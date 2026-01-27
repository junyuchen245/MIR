"""Interface for the composable mapping module."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from MIR.models.SITReg.composable_mapping.coordinate_system import CoordinateSystem


class ICoordinateSystemContainer(ABC):
    """Class holding a unique coordinate system."""

    @property
    @abstractmethod
    def coordinate_system(
        self,
    ) -> "CoordinateSystem":
        """Coordinate system of the container."""
