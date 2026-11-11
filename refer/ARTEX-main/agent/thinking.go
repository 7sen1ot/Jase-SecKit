package agent

import (
	"context"
	"errors"
	"iter"
	"log"
	"strings"
	"sync/atomic"

	"github.com/Autumn-27/norma/llm"
)

// ThinkingParameterError is a deterministic model/configuration mismatch. It
// must not cause the task router or worker to replay an entire agent execution.
type ThinkingParameterError struct{ Cause error }

func (e *ThinkingParameterError) Error() string {
	return "model thinking parameters: " + e.Cause.Error()
}
func (e *ThinkingParameterError) Unwrap() error { return e.Cause }

func IsThinkingParameterError(err error) bool {
	var target *ThinkingParameterError
	return errors.As(err, &target)
}

// Infer only from an explicit capability rejection, never a model-name guess or
// a generic 400. Some compatible endpoints return this message in an SSE error
// without preserving the numeric error code.
func thinkingRequiredError(err error) bool {
	if err == nil {
		return false
	}
	s := strings.ToLower(err.Error())
	return strings.Contains(s, "该模型始终思考，不支持关闭思考") ||
		strings.Contains(s, "does not support disabling thinking") ||
		strings.Contains(s, "thinking cannot be disabled")
}

func classifyThinkingError(err error) error {
	if err == nil || IsThinkingParameterError(err) {
		return err
	}
	s := strings.ToLower(err.Error())
	parameter := strings.Contains(s, "thinking") || strings.Contains(s, "reasoning") ||
		strings.Contains(s, "思考") || strings.Contains(s, "output_config.effort")
	rejected := thinkingRequiredError(err) || strings.Contains(s, "unsupported_parameter") ||
		strings.Contains(s, "invalid_request_error") || strings.Contains(s, "invalid_value") ||
		strings.Contains(s, "not supported") || strings.Contains(s, "不支持")
	if parameter && rejected {
		return &ThinkingParameterError{Cause: err}
	}
	return err
}

type lowThinkingKey struct{}

// thinkingProvider learns capabilities per provider instance (endpoint, key and
// model). The shared SDK provider, limiter and normal request options stay intact.
// Both ARTEX's graph compactor and Norma's session summarizer pass through here.
type thinkingProvider struct {
	llm.Provider
	model        string
	thinkingType string
	required     atomic.Bool
}

func (p *thinkingProvider) disabled(req llm.CompletionRequest) bool {
	mode := req.Thinking
	if mode == "" {
		mode = p.thinkingType
	}
	return mode == "disabled"
}

func (p *thinkingProvider) prepare(ctx context.Context, req llm.CompletionRequest) (context.Context, llm.CompletionRequest) {
	if p.required.Load() && p.disabled(req) {
		req.Thinking = "enabled"
		// A disabled-thinking summary used to reserve only 1500 output tokens.
		// Mandatory reasoning shares that budget; allow the SDK's normal 8192
		// default so reasoning does not consume the entire visible summary.
		if req.MaxTokens > 0 && req.MaxTokens < 8192 {
			req.MaxTokens = 8192
		}
		ctx = context.WithValue(ctx, lowThinkingKey{}, true)
	}
	return ctx, req
}

func (p *thinkingProvider) learn(err error) bool {
	if !thinkingRequiredError(err) {
		return false
	}
	if p.required.CompareAndSwap(false, true) {
		log.Printf("[llm-thinking] model=%s capability=thinking_required source=explicit_rejection fallback=enabled/low retries=1", p.model)
	}
	return true
}

func (p *thinkingProvider) Complete(ctx context.Context, req llm.CompletionRequest) (llm.Message, string, llm.Usage, error) {
	callCtx, callReq := p.prepare(ctx, req)
	msg, stop, usage, err := p.Provider.Complete(callCtx, callReq)
	if p.learn(err) && p.disabled(callReq) && ctx.Err() == nil {
		callCtx, callReq = p.prepare(ctx, req)
		msg, stop, usage, err = p.Provider.Complete(callCtx, callReq)
	}
	return msg, stop, usage, classifyThinkingError(err)
}

func (p *thinkingProvider) Stream(ctx context.Context, req llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		for attempt := 0; attempt < 2; attempt++ {
			callCtx, callReq := p.prepare(ctx, req)
			delivered, retry := false, false
			for ev, err := range p.Provider.Stream(callCtx, callReq) {
				if err != nil {
					if p.learn(err) && p.disabled(callReq) && !delivered && attempt == 0 && ctx.Err() == nil {
						retry = true
						break
					}
					yield(ev, classifyThinkingError(err))
					return
				}
				delivered = true
				if !yield(ev, nil) {
					return
				}
			}
			if !retry {
				return
			}
		}
	}
}
