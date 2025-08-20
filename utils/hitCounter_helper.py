from collections import defaultdict
import asyncio

class HitCounter:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.total_hits = 0
        self.failed_hits = 0
        self.field_counts = defaultdict(int)  # key: fields parsed (1–6)

    async def log_hit(self, nid_data: dict):
        if not isinstance(nid_data, dict):
            return 
        if nid_data.get("nid_data"):
            fields_parsed = sum(1 for v in nid_data['nid_data'].values() if v and str(v).strip())
            async with self.lock:
                self.total_hits += 1
                if fields_parsed == 0:
                    self.failed_hits += 1
                elif 1 <= fields_parsed <= 6:
                    self.field_counts[fields_parsed] += 1

    async def get_summary(self):
        async with self.lock:
            if self.total_hits == 0:
                return {
                    "total_hits": 0,
                    "failed_hits": 0,
                    "avg_fields_per_successful_hit": 0.0,
                    "success_details": {}
                }

            total_fields = sum(fields * count for fields, count in self.field_counts.items())
            successful_hits = self.total_hits - self.failed_hits
            avg = total_fields / successful_hits if successful_hits > 0 else 0.0

            return {
                "total_hits": self.total_hits,
                "failed_hits": self.failed_hits,
                "avg_fields_per_successful_hit": round(avg, 2),
                "success_details": dict(self.field_counts)
            }
