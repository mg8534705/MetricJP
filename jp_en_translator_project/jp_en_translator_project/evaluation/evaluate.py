import sacrebleu

class Evaluator:
    def __init__(self):
        self.translations = []
        self.references = []
        self.token_timings = []

    def add_sample(self, reference, translation, token_timestamps):
        self.references.append(reference)
        self.translations.append(translation)
        self.token_timings.append(token_timestamps)

    def compute_bleu(self):
        return sacrebleu.corpus_bleu(self.translations, [self.references]).score

    def compute_latency(self):
        al_sum = 0
        atd_sum = 0
        count = 0
        for timestamps in self.token_timings:
            if not timestamps:
                continue
            src_len = len(timestamps)
            delays = [t[1] - t[0] for t in timestamps]
            al_sum += sum(delays) / src_len
            atd_sum += max(delays)
            count += 1
        avg_al = al_sum / count if count else 0
        avg_atd = atd_sum / count if count else 0
        return avg_al, avg_atd

    def report(self):
        bleu = self.compute_bleu()
        al, atd = self.compute_latency()
        print(f"BLEU Score: {bleu:.2f}")
        print(f"Average Lagging (AL): {al:.3f} sec")
        print(f"Average Token Delay (ATD): {atd:.3f} sec")
