from typing import Generator, Any
from PIL import Image
from torch import Tensor

type Generator_Image_Range = Generator[tuple[int, int, Image.Image], None, None]

type Generator_Generic_Range = Generator[tuple[int, int, Any], None, None]

type Generator_Range = Generator[tuple[int, int], None, None]

type Generator_Batch_Image_Range = Generator[tuple[list[Image.Image], list[tuple[int, int]]], None, None]

type Generator_Batch_Tensor_Range = Generator[tuple[Tensor, list[tuple[int, int]]], None, None]