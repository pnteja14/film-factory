class FilmFactoryException(Exception):
    """Base exception for Film Factory pipeline."""
    pass

class AgentException(FilmFactoryException):
    """Base exception for all agents."""
    pass

class StoryArchitectException(AgentException):
    """Story Architect specific exception."""
    pass

class ScreenplayWriterException(AgentException):
    """Screenplay Writer specific exception."""
    pass

class SceneDirectorException(AgentException):
    """Scene Director specific exception."""
    pass

class VoiceoverAgentException(AgentException):
    """Voiceover Agent specific exception."""
    pass

class ImageGeneratorException(AgentException):
    """Image Generator specific exception."""
    pass

class ValidationException(FilmFactoryException):
    """Data validation failed."""
    pass

class APIException(FilmFactoryException):
    """External API call failed."""
    pass

class RetryableException(APIException):
    """Exception that can be safely retried."""
    pass

class PermanentException(APIException):
    """Exception that should not be retried."""
    pass