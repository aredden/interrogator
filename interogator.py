import clip
import gc
import pandas as pd
import requests
import sys
from PIL import Image
from io import BytesIO

sys.path.append("./BLIP")
import torch
import pydash as p_
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from BLIP.models.blip import BLIP_Decoder, blip_decoder
from torch.nn.functional import cosine_similarity
from pydantic import BaseModel, BaseConfig, BaseSettings, Field
from typing_extensions import Literal
from typing import List, Dict, Any, Optional, Union, Tuple, Callable, cast

# from clip.model import CLIP
from open_clip import create_model_and_transforms as create_clip, CLIP, tokenize
from clip_options import ClipOptions
import pydash as p_


class RuntimeConfig(BaseModel):
    device: Optional[torch.device] = Field(
        default_factory=lambda: torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
    )
    blip_image_eval_size: int = 384
    clip_models: List[Tuple[CLIP, T.Compose]] = Field(default_factory=lambda: [])
    blip_model_size: Literal["base", "large"] = "base"
    in_colab: bool = False
    blip_model: Optional[BLIP_Decoder] = None
    transform: Optional[T.Compose] = None

    class Config(BaseConfig):
        arbitrary_types_allowed: bool = True


def load_clip(
    name_type: Tuple[str, str],
    enable_checkpointing: bool = True,
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
) -> Tuple[CLIP, T.Compose]:
    model, _, preprocess = create_clip(name_type[0], name_type[1], "amp", device=device)
    model: CLIP = cast(CLIP, model)
    if enable_checkpointing:
        model.set_grad_checkpointing(True)
    model = model.half().eval().to(device)
    return model, preprocess


def load_blip(
    model_size: Literal["base", "large"] = "base",
    eval_size: int = 384,
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
) -> Tuple[BLIP_Decoder, T.Compose]:
    if model_size == "base":
        model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth"
    elif model_size == "large":
        model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption.pth"
    else:
        raise ValueError(f"Model size {model_size} is not supported")
    blip_model = blip_decoder(
        pretrained=model_url, image_size=eval_size, vit=model_size
    )
    blip_model.eval()
    blip_model = blip_model.to(device)
    transform = T.Compose(
        [
            T.Resize(
                (eval_size, eval_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.ToTensor(),
            T.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    return blip_model, transform


class Interrogator:
    def __init__(self, runcfg: RuntimeConfig) -> None:
        self.runcfg: RuntimeConfig = runcfg
        self.device: torch.device = runcfg.device
        if self.runcfg.blip_model is None:
            self.blip_model, self.transform = load_blip(
                runcfg.blip_model_size, runcfg.blip_image_eval_size, self.device
            )
        else:
            self.blip_model: BLIP_Decoder = runcfg.blip_model
            self.transform: T.Compose = runcfg.transform
        self.attrs: Attrs = Attrs()
        self.in_colab = runcfg.in_colab

    def generate_caption(self, pil_image) -> str:
        gpu_image = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            caption = self.blip_model.generate(
                gpu_image, sample=False, num_beams=3, max_length=20, min_length=5
            )
        return caption[0]

    def generate(self, image, models, batch_size=1024, top_count=1, from_each=10):
        caption = self.generate_caption(image)
        if len(models) == 0:
            return caption

        table = []
        bests = [[("", 0)]] * 5
        bests_adjs = []
        for model_name in models:
            print(f"Interrogating with {model_name}...")
            model, preprocess = load_clip(
                model_name, enable_checkpointing=True, device=self.device
            )
            images: torch.Tensor = preprocess(image).unsqueeze(0).to(self.device).half()
            with torch.no_grad():
                image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            ranks = [
                self.rank(model, image_features, self.attrs.mediums),
                self.rank(
                    model,
                    image_features,
                    ["by " + artist for artist in self.attrs.artists],
                ),
                self.rank(model, image_features, self.attrs.trending_list),
                self.rank(model, image_features, self.attrs.movements),
                self.rank(model, image_features, self.attrs.flavors, top_count=3),
            ]
            adj_ranks = self.rank_adj(
                model,
                image_features,
                self.attrs.adjectives,
                top_count=5,
                batch_size=batch_size,
                from_each=from_each,
            )
            bests_adjs.extend(adj_ranks)
            bests_adjs = p_.uniq_with(bests_adjs, comparator=lambda a, b: a[0] == b[0])
            bests_adjs = sorted(bests_adjs, key=lambda x: x[1], reverse=True)

            for i in range(len(ranks)):
                confidence_sum = 0
                for ci in range(len(ranks[i])):
                    confidence_sum += ranks[i][ci][1]
                if confidence_sum > sum(bests[i][t][1] for t in range(len(bests[i]))):
                    bests[i] = ranks[i]

            row = [model_name]
            for r in ranks:
                row.append(", ".join([f"{x[0]} ({x[1]:0.1f}%)" for x in r]))

            table.append(row)

            model = model.cpu()
            torch.cuda.synchronize(self.device)
            del model
            gc.collect()
            df: pd.DataFrame = pd.DataFrame(
                table,
                columns=[
                    "Model",
                    "Medium",
                    "Artist",
                    "Trending",
                    "Movement",
                    "Flavors",
                ],
            )
            if self.in_colab:
                display(df.head(2))  # type: ignore
            else:
                print(df.head(2))

        # bests_adjs = list(set([b[0] for b in bests_adjs]))
        bests_adjs = p_.uniq_with(bests_adjs, lambda a, b: a[0] == b[0])

        flaves: str = ", ".join([f"{x[0]}" for x in bests[4]])
        medium: str = bests[0][0][0]
        medium_middle = ""
        if caption.startswith(medium):
            medium_middlle = f", {medium} "

        adjective_beginning = (
            f"A {bests_adjs[0][0].lower()}, {bests_adjs[1][0].lower()} "
        )
        caption = f"{caption if not caption.lower().startswith('a ') else caption[2:]}"
        artist = f" {bests[1][0][0]}, "
        styles = (
            f"trending on {bests[2][0][0]}, in the style of {bests[3][0][0]}, {flaves}"
        )
        result_string = f"{adjective_beginning}{caption}{artist}{medium_middle}{styles}"
        return result_string.strip()

    def rank_adj(
        self,
        model: CLIP,
        image_features: torch.Tensor,
        text_array: List[str],
        top_count: int = 1,
        batch_size: int = 1024,
        from_each: int = 10,
    ) -> List[Tuple[str, float]]:
        top_count = min(top_count, len(text_array))
        ranks = []
        for idx, text_chunk in enumerate(chunk(text_array, batch_size)):
            text_tokens = tokenize([text for text in text_chunk]).to(self.device)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens).half()
                scores = cosine_similarity(image_features.half(), text_features)

            ranks.extend(
                [
                    [text_chunk[i], scores[i].cpu().numpy().item()]
                    for i in torch.argsort(scores, descending=True)[:from_each]
                ]
            )

        return sorted(ranks, key=lambda x: x[1], reverse=True)[:top_count]

    def rank(
        self,
        model: CLIP,
        image_features: torch.Tensor,
        text_array: List[str],
        top_count=1,
    ) -> List[Tuple[str, float]]:
        top_count = min(top_count, len(text_array))

        text_tokens: torch.Tensor = tokenize([text for text in text_array]).to(
            self.device
        )
        with torch.no_grad():
            text_features: torch.Tensor = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity: torch.Tensor = torch.cosine_similarity(
            image_features, text_features, dim=-1
        )

        top_probs, top_labels = similarity.float().cpu().topk(top_count, dim=-1)
        return [
            (text_array[top_labels[i].numpy()], (top_probs[i].numpy() * 100))
            for i in range(top_count)
        ]


def load_list(name):
    with open(f"clip_data/{name}.txt", "r", encoding="utf-8", errors="replace") as f:
        items = [line.strip() for line in f.readlines()]
    return items


def chunk(items, chunk_size):
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def build_trending_list(trending_list: List[str]):
    trending_list_copy = p_.clone_deep(trending_list)
    trending_list += ["trending on " + site for site in trending_list_copy]
    trending_list += ["featured on " + site for site in trending_list_copy]
    trending_list += [site + " contest winner" for site in trending_list_copy]
    return trending_list


class Attrs(BaseModel):
    artists: List[str] = Field(default_factory=lambda: load_list("artists"))
    flavors: List[str] = Field(default_factory=lambda: load_list("flavors"))
    mediums: List[str] = Field(default_factory=lambda: load_list("mediums"))
    movements: List[str] = Field(default_factory=lambda: load_list("movements"))
    adjectives: List[str] = Field(default_factory=lambda: load_list("adjectives"))
    sites: List[str] = Field(default_factory=lambda: load_list("sites"))
    trending_list: List[str] = Field(
        default_factory=lambda: build_trending_list(load_list("sites"))
    )


if __name__ == "__main__":

    image_path_or_url: str = "/home/alex/Documents/tokenly/pixelmind-generator/latent-retrieval/3116627661_1_5_yyRypRhb.jpg"
    if image_path_or_url.startswith("http"):
        image: Image.Image = Image.open(
            BytesIO(requests.get(image_path_or_url).content)
        ).convert("RGB")
    else:
        image: Image.Image = Image.open(image_path_or_url).convert("RGB")

    model_opts = [
        ClipOptions.ViT_L_14_336_openai,
        ClipOptions.ViT_B_16_plus_240_laion400m_e32,
    ]
    runcfg = RuntimeConfig()
    thumb: Image.Image = image.copy()
    svc = Interrogator(runcfg=runcfg)

    result = svc.generate(thumb, model_opts, batch_size=256)

    print(result)
