from typing import Optional, List
import numpy as np

class BaseModel:
    @classmethod
    def build(cls) -> "BaseModel":
        """Build the model."""
        raise NotImplementedError

    def get_answer(
        self,
        image: np.ndarray,
        question: str,
        answer_candidates: Optional[List[str]] = None,
    ) -> str:
        """Get the answer to the question given the image.

        :param image: the image to be processed in numpy array format
            of shape (H, W, C) in RGB order
        :param question: the question to be answered in string format
        :param answer_candidates: the answer candidates for the question to prevent
            the model from generating irrelevant answers. For example, the answer
            candidates for "SameDiffQA" would be ["yes", "no"], which can be used
            to either prompt the model or for parsing the model output.
        :return: the answer to the question in string format
        """
        raise NotImplementedError

    def get_box(self, image: np.ndarray, query: str) -> List[float]:
        """Get the bounding box of the object in the image that is
        described by the query.

        :param image: the image to be processed in numpy array format
            of shape (H, W, C) in RGB order
        :param query: the query to be answered in string format
        :return: the bounding box of the object in the image that is
            described by the query in list format [x1, y1, x2, y2]
        """
        raise NotImplementedError

class CustomModel(BaseModel):
    pass