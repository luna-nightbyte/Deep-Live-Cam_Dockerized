from insightface.app.common import Face
from typing import Any
import numpy


Face = Face
Frame = numpy.ndarray[Any, Any]

DetectedFace = dict  # Define a type alias for detected objects.  Use a dictionary
