from typing import Generator

type Detection_Range_Count = tuple[dict[str, int], tuple[int, int]]

type Generator_Range_Detection_Count = Generator[Detection_Range_Count, None, None]