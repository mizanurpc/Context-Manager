import tiktoken
import random

class ContextManager:
    def __init__(self, window_size=8000, budgets=None):
        self.window_size = window_size
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.budgets = budgets or {
            "system": 600,
            "rag": 3000,
            "history": 3000,
            "reply": 1400
        }
        assert sum(self.budgets.values()) <= self.window_size, "Budget exceeds window size!"

    def ntok(self, text):
        return len(self.enc.encode(text))

    def clip_at_sentence(self, text, cap):
        """ Hard cap and clip at sentence boundaries where possible """
        ids = self.enc.encode(text)
        if len(ids) <= cap:
            return text
        clipped_ids = ids[:cap]
        decoded = self.enc.decode(clipped_ids)
        # Try to clip neatly at the last full sentence boundary if possible
        last_punct = max(decoded.rfind('.'), decoded.rfind('?'), decoded.rfind('!'))
        if last_punct != -1:
            decoded = decoded[:last_punct + 1]
        return decoded

    def compress_overflow(self, old_turns, summariser):
        """ Summarise overflow instead of dropping it """
        text = "\n".join(f"{m['role']}: {m['content']}" for m in old_turns)
        summary = summariser(f"Summarise for context, <400 tokens:\n{text}")
        return {"role": "system", "content": f"[earlier summary] {summary}"}

    def build_context(self, messages, rag_text, summariser):
        """ Requirement 1, 2 & 3: Keeps system, reply room, applies budgets & summarises overflow """
        system_msg = messages[0]
        rest_messages = messages[1:]
        
        # Clip RAG based on budget
        clipped_rag = self.clip_at_sentence(rag_text, self.budgets["rag"])
        
        # Handle history sliding window & overflow compression
        history_budget = self.budgets["history"]
        kept = []
        overflow = []
        current_used = 4  # overhead
        
        for m in reversed(rest_messages):
            cost = self.ntok(m["content"]) + 4
            if current_used + cost > history_budget:
                overflow.insert(0, m)  # older turns go to overflow
            else:
                kept.insert(0, m)
                current_used += cost

        # Summarise overflow if any exists
        context_messages = [system_msg]
        if overflow:
            summary_msg = self.compress_overflow(overflow, summariser)
            context_messages.append(summary_msg)
            
        context_messages.extend(kept)
        
        # Calculate before/after tokens for demonstration
        orig_tokens = sum(self.ntok(m["content"]) for m in messages) + self.ntok(rag_text)
        new_tokens = sum(self.ntok(m["content"]) for m in context_messages) + self.ntok(clipped_rag)
        
        print(f"Token Count -> Before: {orig_tokens} tokens | After: {new_tokens} tokens")
        return context_messages, clipped_rag

    def run_needle_test(self, lab_model_func, n_values=[100, 500], positions=None):
        """ Requirement 4: Needle test at 3 positions x 2 context sizes """
        NEEDLE = "The vault code for lab-7 is TOPAZ-42."
        QUESTION = "What is the vault code for lab-7?"
        filler = "The sky is blue and the grass is green. Security logs are nominal."
        
        results = []
        for n in n_values:
            pos_list = [6, n // 2, n - 1]
            for pos in pos_list:
                lines = [filler] * n
                lines[pos] = NEEDLE
                doc = "\n".join(f"{i}: {l}" for i, l in enumerate(lines))
                
                answer = lab_model_func(doc, pos, n)
                found = "TOPAZ-42" in answer
                results.append({"Context Size (n)": n, "Position": pos, "Found": found})
        return results

# --- Test Execution ---
if __name__ == "__main__":
    cm = ContextManager()
    
    sample_messages = [
        {"role": "system", "content": "You are a secure assistant."},
        {"role": "user", "content": "Hello, let's start the pentest."},
        {"role": "assistant", "content": "Understood, ready."},
        {"role": "user", "content": "Analyze log entry A."},
        {"role": "assistant", "content": "Log entry A looks clean."},
        {"role": "user", "content": "Analyze log entry B."},
        {"role": "assistant", "content": "Log entry B has a warning."}
    ]
    
    rag_doc = "Confidential server metrics: CPU load is normal. Database latency is 12ms."
    
    def dummy_summariser(prompt):
        return "Previous turns discussed initial pentest setup and log entries A and B."
    
    print("--- Building Context ---")
    ctx, rag = cm.build_context(sample_messages, rag_doc, dummy_summariser)
    
    print("\n--- Running Needle Test ---")
    def mock_lab_model(doc, pos, n):
        # Simulate 'Lost in the Middle' failure when position is right at the center for large contexts
        if n > 100 and pos == n // 2:
            return "I cannot find the code."
        return "The vault code for lab-7 is TOPAZ-42."
        
    needle_results = cm.run_needle_test(mock_lab_model)
    for r in needle_results:
        print(r)