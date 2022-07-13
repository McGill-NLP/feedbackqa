from abc import ABC, abstractmethod
from typing import List, NamedTuple


# If you use Python 3.8, TypedDict is preferred instead.
class Passage(NamedTuple):
    uri: str
    source: str
    content: str
    content_html: str


class FeedbackSystemABC(ABC):
    @abstractmethod
    def build_knowledge_base(self, passages: List[Passage]) -> None:
        """
        You should use this method to create a knowledge base containing all
        the passages that might be relevant to a future query. The passages must have
        a certain structure in order to be used inside the GUI.

        PARAMETERS
            passages:
                A list of passages, which are dictionaries with specific keys (see
                the "Passage" named tuple as a reference).
        """
        pass

    def load_kb_files(self, load_dir: str, verbose: bool = False) -> List[Passage]:
        """
        This is an optional helper function for generating the passages that will
        be be given to the build_knowledge_base method.

        PARAMETERS
            load_dir:
                The directory where we are loading JSON files containing each groups

        RETURNS
            passages:
                A list of passages, which are dictionaries with specific keys (see
                the "Passage" named tuple as a reference).
        """
        pass

    def retrieve_idx(self, query: str, k: int = 10) -> List[int]:
        """
        An optional helper function for retrieving the indices of the most relevant passages.
        This can be inside the retrieve function.

        PARAMETERS
            query:
                The question for which you would like to retrieve passages containing
                possible answers
            k:
                The number of indices that you want to retrieve


        RETURNS
            A list of indices, sorted by the order of relevance (from most relevant) to
            least relevant.
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, k: int = 10) -> List[dict]:
        """
        Given a query, retrieves the most relevant passages from the internal KB.

        PARAMETERS
            query:
                The question for which you would like to retrieve passages containing
                possible answers
            k:
                The number of indices that you want to retrieve


        RETURNS
            A list of passages, sorted by the order of relevance (from most relevant) to
            least relevant.
        """
        pass

    @abstractmethod
    def rate(self, query: str, candidates: List[str]) -> List[str]:
        pass

    @abstractmethod
    def give_feedback(
        self, query: str, candidates: List[str], verbose: bool = True
    ) -> List[str]:
        pass
