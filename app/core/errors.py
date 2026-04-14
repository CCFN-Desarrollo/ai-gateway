class AIGatewayError(Exception):
    """Base exception for controlled gateway failures."""


class RuntimeConfigurationError(AIGatewayError):
    """Raised when required runtime configuration is missing or invalid."""


class UpstreamServiceError(AIGatewayError):
    """Raised when an upstream provider cannot be reached reliably."""


class ProviderResponseError(AIGatewayError):
    """Raised when an upstream provider returns an invalid response shape."""


class CRMClientError(AIGatewayError):
    """Raised when the CRM integration fails."""


class CRMValidationError(CRMClientError):
    """Raised when the CRM rejects the payload as invalid or incomplete."""


class CRMConflictError(CRMClientError):
    """Raised when the CRM reports the client is already in review."""
