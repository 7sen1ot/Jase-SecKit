package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/Autumn-27/norma/llm"
)

const requiredThinkingResponse = `{"type":"error","error":{"type":"invalid_request_error","code":"1210","message":"该模型始终思考，不支持关闭思考；请使用 low、high 或 max。"}}`
const thinkingOKResponse = `{"type":"message","role":"assistant","content":[{"type":"text","text":"OK"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`

func TestThinkingCapabilityWire(t *testing.T) {
	var mu sync.Mutex
	var bodies []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		mu.Lock()
		bodies = append(bodies, body)
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if body["thinking"].(map[string]any)["type"] == "disabled" {
			w.WriteHeader(400)
			fmt.Fprint(w, requiredThinkingResponse)
			return
		}
		fmt.Fprint(w, thinkingOKResponse)
	}))
	defer srv.Close()
	c := ConfigFrom("anthropic", "arbitrary-model-alias", srv.URL, "test", "")
	c.ThinkingType, c.ReasoningEffort = "enabled", "max"
	prov, err := c.NewProvider()
	if err != nil {
		t.Fatal(err)
	}
	req := llm.CompletionRequest{Messages: []llm.Message{llm.UserText("OK")}, Thinking: "disabled", MaxTokens: 1500}
	for i := 0; i < 2; i++ {
		msg, _, _, err := prov.Complete(context.Background(), req)
		if err != nil || msg.Text() != "OK" {
			t.Fatalf("summary %d: %q %v", i, msg.Text(), err)
		}
	}
	if _, _, _, err := prov.Complete(context.Background(), llm.CompletionRequest{Messages: req.Messages}); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(bodies) != 4 {
		t.Fatalf("requests=%d, want rejection, adapted, cached, main", len(bodies))
	}
	for _, i := range []int{1, 2} {
		if bodies[i]["thinking"].(map[string]any)["type"] != "enabled" ||
			bodies[i]["output_config"].(map[string]any)["effort"] != "low" || bodies[i]["max_tokens"] != float64(8192) {
			t.Fatalf("adapted body: %#v", bodies[i])
		}
	}
	if bodies[3]["output_config"].(map[string]any)["effort"] != "max" {
		t.Fatal("main agent effort changed")
	}
}

func TestThinkingOptionalModelUnchanged(t *testing.T) {
	for _, effort := range []string{"low", "high", "max"} {
		t.Run(effort, func(t *testing.T) {
			count := 0
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				count++
				var body map[string]any
				_ = json.NewDecoder(r.Body).Decode(&body)
				if body["thinking"].(map[string]any)["type"] != "disabled" || body["output_config"].(map[string]any)["effort"] != effort || body["max_tokens"] != float64(1500) {
					t.Errorf("optional model changed: %#v", body)
				}
				fmt.Fprint(w, thinkingOKResponse)
			}))
			defer srv.Close()
			c := ConfigFrom("anthropic", "same-model-name", srv.URL, "test", "")
			c.ReasoningEffort = effort
			p, err := c.NewProvider()
			if err != nil {
				t.Fatal(err)
			}
			_, _, _, err = p.Complete(context.Background(), llm.CompletionRequest{Thinking: "disabled", MaxTokens: 1500})
			if err != nil || count != 1 {
				t.Fatalf("requests=%d err=%v", count, err)
			}
		})
	}
}

func TestThinkingUnsupportedEffortNoRetry(t *testing.T) {
	count := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count++
		w.WriteHeader(400)
		fmt.Fprint(w, `{"error":{"type":"invalid_request_error","message":"unsupported_parameter: output_config.effort"}}`)
	}))
	defer srv.Close()
	c := ConfigFrom("anthropic", "unknown", srv.URL, "test", "")
	c.ThinkingType, c.ReasoningEffort = "enabled", "xhigh"
	p, err := c.NewProvider()
	if err != nil {
		t.Fatal(err)
	}
	_, _, _, err = p.Complete(context.Background(), llm.CompletionRequest{})
	if !IsThinkingParameterError(err) || count != 1 {
		t.Fatalf("requests=%d err=%v", count, err)
	}
}

func TestThinkingStreamReplayBoundary(t *testing.T) {
	for _, partial := range []bool{false, true} {
		t.Run(fmt.Sprint(partial), func(t *testing.T) {
			count := 0
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				count++
				w.Header().Set("Content-Type", "text/event-stream")
				if count == 1 {
					if partial {
						fmt.Fprint(w, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"part\"}}\n\n")
					}
					fmt.Fprintf(w, "event: error\ndata: %s\n\n", requiredThinkingResponse)
					return
				}
				fmt.Fprint(w, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"OK\"}}\n\n")
			}))
			defer srv.Close()
			c := ConfigFrom("anthropic", "unknown", srv.URL, "test", "")
			p, err := c.NewProvider()
			if err != nil {
				t.Fatal(err)
			}
			var result string
			var streamErr error
			for ev, err := range p.Stream(context.Background(), llm.CompletionRequest{Thinking: "disabled"}) {
				result += ev.Text
				if err != nil {
					streamErr = err
				}
			}
			if partial {
				if count != 1 || result != "part" || !IsThinkingParameterError(streamErr) {
					t.Fatalf("replayed partial stream: %d %q %v", count, result, streamErr)
				}
			} else if count != 2 || result != "OK" || streamErr != nil {
				t.Fatalf("initial rejection not adapted: %d %q %v", count, result, streamErr)
			}
		})
	}
}
