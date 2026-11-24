package server

import (
	"errors"
	"fmt"
	"testing"

	"github.com/Autumn-27/artex/agent"
	"github.com/Autumn-27/norma/harness"
)

func TestThinkingParameterErrorNeverReplaysWorker(t *testing.T) {
	err := fmt.Errorf("wrapped: %w", &agent.ThinkingParameterError{Cause: errors.New("unsupported reasoning_effort")})
	if isRetryableStreamError(err) || retryableWorkerModelError(harness.ReasonModelError, err) {
		t.Fatal("deterministic thinking error must not replay model request or whole worker")
	}
	if !isRetryableStreamError(errors.New("connection reset by peer")) ||
		!retryableWorkerModelError(harness.ReasonModelError, errors.New("connection reset by peer")) {
		t.Fatal("ordinary transient error retry behavior changed")
	}
}
