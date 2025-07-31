import redis
import os
import json
from datetime import datetime

# Use the same Redis URL as your other scripts
REDIS_URL = 'rediss://red-d0mvcmd6ubrc73epattg:pyQbOXLbZn7yNczcJhQ9MCHYoeKR4045@ohio-keyvalue.render.com:6379'
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

def probe_redis():
    print("Probing Redis database...")
    try:
        keys = list(redis_client.scan_iter('*'))
    except redis.exceptions.ResponseError as e:
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
                value = redis_client.get(key)
                if value is None:
                    value = redis_client.lrange(key, 0, -1)  # Get all list items, not just first 10
                
                print(f"Key: {key}")
                dump_file.write(f"Key: {key}\n")
                
                if isinstance(value, list):
                    print(f"  Type: list, Length: {len(value)}")
                    dump_file.write(f"  Type: list, Length: {len(value)}\n")
                    
                    for i, item in enumerate(value[:3]):
                        try:
                            parsed = json.loads(item)
                            formatted_json = json.dumps(parsed, indent=2)
                            print(f"    [{i}] {formatted_json}")
                            dump_file.write(f"    [{i}] {formatted_json}\n")
                        except Exception:
                            print(f"    [{i}] {item}")
                            dump_file.write(f"    [{i}] {item}\n")
                    
                    if len(value) > 3:
                        print("    ...")
                        dump_file.write("    ...\n")
                    
                    # Write all items to dump file for complete backup
                    dump_file.write("  FULL LIST CONTENTS:\n")
                    for i, item in enumerate(value):
                        dump_file.write(f"    [{i}] {item}\n")
                        
                else:
                    try:
                        parsed = json.loads(value)
                        formatted_json = json.dumps(parsed, indent=2)
                        print(f"  Type: string, JSON: {formatted_json}")
                        dump_file.write(f"  Type: string, JSON: {formatted_json}\n")
                    except Exception:
                        print(f"  Type: string, Value: {str(value)}")
                        dump_file.write(f"  Type: string, Value: {str(value)}\n")
                
                dump_file.write("\n" + "-"*40 + "\n\n")
                        
            except Exception as e:
                error_msg = f"  Error reading key {key}: {e}"
                print(error_msg)
                dump_file.write(error_msg + "\n\n")
    
    print(f"\nComplete Redis dump saved to: {dump_filename}")

if __name__ == "__main__":
    probe_redis()
