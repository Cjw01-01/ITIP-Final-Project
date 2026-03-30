from qdrant_client import QdrantClient

qb = QdrantClient(url="http://localhost:6334")
info = qb.get_collection("candidate_profiles")
print(f"Total points in candidate_profiles: {info.points_count}")
print()

r = qb.scroll(collection_name="candidate_profiles", limit=200, with_payload=True)
zara = [
    p for p in r[0]
    if "zara" in (p.payload.get("candidate_name", "") or "").lower()
    or "nakamura" in (p.payload.get("source_file", "") or "").lower()
    or "nakamura" in (p.payload.get("text", "") or "").lower()
]
print(f"Zara entries found: {len(zara)}")
for p in zara:
    sid = p.payload.get("source_id", "?")
    name = p.payload.get("candidate_name", "?")
    track = p.payload.get("bmw_track_label", "?")
    skills = p.payload.get("skills", "NONE")
    text = (p.payload.get("text", "") or "")[:200]
    print(f"  source_id: {sid}")
    print(f"  candidate_name: {name}")
    print(f"  bmw_track_label: {track}")
    print(f"  skills: {skills}")
    print(f"  text: {text}...")
    print()

if not zara:
    print("Zara is NOT in the database. The upload may have failed.")
    print("Check Agent A logs for errors.")
