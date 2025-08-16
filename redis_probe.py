import redis
import os
import json
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple, cast
from redis.exceptions import ResponseError

# Use the same Redis URL as your other scripts
REDIS_URL = 'rediss://red-d0mvcmd6ubrc73epattg:pyQbOXLbZn7yNczcJhQ9MCHYoeKR4045@ohio-keyvalue.render.com:6379'
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

def probe_redis():
    print("Probing Redis database...")
    try:
        keys = list(redis_client.scan_iter('*'))
    except ResponseError as e:
        if "allowlist" in str(e) or "not in the allowlist" in str(e):
            print("ERROR: Your client/server IP is not in the Redis allowlist.")
            print("Go to your Redis/Valkey provider dashboard and add this server's public IP to the allowlist.")
            print("Full error:", e)
            return
        else:
            raise
    except Exception as e:
        print("ERROR: Could not connect to Redis:", e)
        return

    print(f"Found {len(keys)} keys.")
    
    # Create dump file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dump_filename = f"redis_dump_{timestamp}.txt"
    
    with open(dump_filename, 'w', encoding='utf-8') as dump_file:
        dump_file.write(f"Redis Dump - {datetime.now().isoformat()}\n")
        dump_file.write(f"Total keys found: {len(keys)}\n")
        dump_file.write("="*50 + "\n\n")
        
        for key in keys:
            try:
                # Determine the type of the key to avoid WRONGTYPE errors
                dtype = cast(str, redis_client.type(key))
                print(f"Key: {key}")
                dump_file.write(f"Key: {key}\n")

                # Optional: TTL/info
                try:
                    ttl_any = redis_client.ttl(key)
                    ttl = ttl_any if isinstance(ttl_any, int) else None
                    if ttl is not None and ttl >= 0:
                        dump_file.write(f"  TTL: {ttl}s\n")
                except Exception:
                    pass

                if dtype == 'string':
                    value_any = redis_client.get(key)
                    value = cast(str, value_any)
                    try:
                        parsed = json.loads(value)
                        formatted_json = json.dumps(parsed, indent=2)
                        print(f"  Type: string, JSON: {formatted_json}")
                        dump_file.write(f"  Type: string, JSON: {formatted_json}\n")
                    except Exception:
                        print(f"  Type: string, Value: {str(value)}")
                        dump_file.write(f"  Type: string, Value: {str(value)}\n")

                elif dtype == 'list':
                    items_any = redis_client.lrange(key, 0, -1)
                    items = cast(List[str], items_any)
                    print(f"  Type: list, Length: {len(items)}")
                    dump_file.write(f"  Type: list, Length: {len(items)}\n")
                    # Preview first 3
                    for i, item in enumerate(items[:3]):
                        try:
                            parsed = json.loads(item)
                            formatted_json = json.dumps(parsed, indent=2)
                            print(f"    [{i}] {formatted_json}")
                            dump_file.write(f"    [{i}] {formatted_json}\n")
                        except Exception:
                            print(f"    [{i}] {item}")
                            dump_file.write(f"    [{i}] {item}\n")
                    if len(items) > 3:
                        print("    ...")
                        dump_file.write("    ...\n")
                    # Full dump
                    dump_file.write("  FULL LIST CONTENTS:\n")
                    for i, item in enumerate(items):
                        dump_file.write(f"    [{i}] {item}\n")

                elif dtype == 'set':
                    members_any = redis_client.smembers(key)
                    members = sorted(list(cast(Set[str], members_any)))
                    print(f"  Type: set, Length: {len(members)}")
                    dump_file.write(f"  Type: set, Length: {len(members)}\n")
                    for i, m in enumerate(members[:10]):
                        dump_file.write(f"    [{i}] {m}\n")
                    if len(members) > 10:
                        dump_file.write("    ...\n")
                    # Full dump
                    dump_file.write("  FULL SET CONTENTS:\n")
                    for i, m in enumerate(members):
                        dump_file.write(f"    [{i}] {m}\n")

                elif dtype == 'hash':
                    h_any = redis_client.hgetall(key)
                    h = cast(Dict[str, str], h_any)
                    print(f"  Type: hash, Length: {len(h)}")
                    dump_file.write(f"  Type: hash, Length: {len(h)}\n")
                    # Preview a few fields
                    for i, (hk, hv) in enumerate(list(h.items())[:10]):
                        try:
                            parsed = json.loads(hv)
                            hv_fmt = json.dumps(parsed, indent=2)
                            dump_file.write(f"    {hk}: {hv_fmt}\n")
                        except Exception:
                            dump_file.write(f"    {hk}: {hv}\n")
                    if len(h) > 10:
                        dump_file.write("    ...\n")
                    # Full dump
                    dump_file.write("  FULL HASH CONTENTS:\n")
                    for hk, hv in h.items():
                        dump_file.write(f"    {hk}: {hv}\n")

                elif dtype == 'zset':
                    zitems_any = redis_client.zrange(key, 0, -1, withscores=True)
                    zitems = cast(List[Tuple[str, float]], zitems_any)
                    print(f"  Type: zset, Length: {len(zitems)}")
                    dump_file.write(f"  Type: zset, Length: {len(zitems)}\n")
                    for i, (member, score) in enumerate(zitems[:10]):
                        dump_file.write(f"    [{i}] {member} (score={score})\n")
                    if len(zitems) > 10:
                        dump_file.write("    ...\n")
                    # Full dump
                    dump_file.write("  FULL ZSET CONTENTS:\n")
                    for i, (member, score) in enumerate(zitems):
                        dump_file.write(f"    [{i}] {member} (score={score})\n")

                elif dtype == 'stream':
                    # Fetch a limited number of items to avoid huge dumps
                    try:
                        entries_any = redis_client.xrange(key, '-', '+', count=100)
                    except TypeError:
                        # Older redis-py may not support count kw; fallback
                        entries_any = redis_client.xrange(key, '-', '+')
                    entries = cast(List[Tuple[str, Dict[str, str]]], entries_any)
                    print(f"  Type: stream, Sampled: {len(entries)} entries")
                    dump_file.write(f"  Type: stream, Sampled: {len(entries)} entries\n")
                    for i, (msg_id, fields) in enumerate(entries[:10]):
                        dump_file.write(f"    [{i}] id={msg_id} fields={fields}\n")
                    if len(entries) > 10:
                        dump_file.write("    ...\n")
                    # Do not full-dump streams by default

                else:
                    # Unknown or none
                    print(f"  Type: {dtype}")
                    dump_file.write(f"  Type: {dtype}\n")

                dump_file.write("\n" + "-"*40 + "\n\n")
                
            except Exception as e:
                error_msg = f"  Error reading key {key}: {e}"
                print(error_msg)
                dump_file.write(error_msg + "\n\n")
    
    print(f"\nComplete Redis dump saved to: {dump_filename}")

if __name__ == "__main__":
    probe_redis()
