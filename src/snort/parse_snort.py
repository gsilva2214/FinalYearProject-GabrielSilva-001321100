import pandas as pd
from pathlib import Path

def parse_line(line): # parse a single snort fast alert line
    line = line.strip()
    if not line:
        return None

    # timestamp
    space_idx = line.find("  ")
    if space_idx < 0:
        return None
    ts = line[:space_idx].strip()

    first_star = line.find("[**]")     # SID section to find [gid:sid:rev] Skip past first [**]
    if first_star < 0:
        return None
    after_first_star = first_star + 4

    bracket_open = line.find("[", after_first_star) # Find [gid:sid:rev]
    bracket_close = line.find("]", bracket_open)
    if bracket_open < 0 or bracket_close < 0:
        return None

    gid_sid_rev = line[bracket_open + 1:bracket_close]
    parts = gid_sid_rev.split(":")
    if len(parts) != 3:
        return None

    try:
        gid = int(parts[0])
        sid = int(parts[1])
        rev = int(parts[2])
    except ValueError:
        return None

    msg_start = bracket_close + 1
    second_star = line.find("[**]", msg_start)
    if second_star < 0:
        return None
    msg = line[msg_start:second_star].strip()

    class_marker = "[Classification: "
    class_start = line.find(class_marker, second_star)
    classification = "Unclassified"
    if class_start >= 0:
        class_content_start = class_start + len(class_marker)
        class_end = line.find("]", class_content_start)
        if class_end >= 0:
            classification = line[
                class_content_start:class_end
            ].strip()

    prio_marker = "[Priority: "     # priority
    prio_start = line.find(prio_marker)
    priority = 0
    if prio_start >= 0:
        prio_content_start = prio_start + len(prio_marker)
        prio_end = line.find("]", prio_content_start)
        if prio_end >= 0:
            try:
                priority = int(
                    line[prio_content_start:prio_end].strip()
                )
            except ValueError:
                priority = 0

    proto = None
    src_ip = None
    src_port = None
    dst_ip = None
    dst_port = None

    curly_open = line.find("{")
    curly_close = line.find("}", curly_open) if curly_open >= 0 else -1

    if curly_open >= 0 and curly_close >= 0:
        proto = line[curly_open + 1:curly_close].strip()

        remainder = line[curly_close + 1:].strip()
        arrow = remainder.find(" -> ")
        if arrow >= 0:
            src_part = remainder[:arrow].strip()
            dst_part = remainder[arrow + 4:].strip()
            # parse source
            if ":" in src_part:
                last_colon = src_part.rfind(":")
                src_ip = src_part[:last_colon]
                try:
                    src_port = int(src_part[last_colon + 1:])
                except ValueError:
                    src_ip = src_part
                    src_port = None
            else:
                src_ip = src_part

            # parse destination
            if ":" in dst_part:
                last_colon = dst_part.rfind(":")
                dst_ip = dst_part[:last_colon]
                try:
                    dst_port = int(dst_part[last_colon + 1:])
                except ValueError:
                    dst_ip = dst_part
                    dst_port = None
            else:
                dst_ip = dst_part

    return {
        "timestamp": ts,
        "sid": sid,
        "priority": priority,
        "classification": classification,
        "message": msg,
        "protocol": proto,
        "src_ip": src_ip,
        "src_port": src_port,
        "dst_ip": dst_ip,
        "dst_port": dst_port,
    }

def parse_snort_alerts(
    input_path: str,
    output_csv: str,
) -> str:

    data_dir = Path(input_path)
    rows = []
    total_lines = 0
    matched = 0
    with_ips = 0

    for log_file in sorted(data_dir.glob("alert_*.log")):
        day = log_file.stem.replace("alert_", "")
        file_count = 0

        with log_file.open(errors="ignore") as f:
            for line in f:
                total_lines += 1
                result = parse_line(line)
                if result is None:
                    continue

                result["day"] = day
                matched += 1
                file_count += 1

                if result["src_ip"] is not None:
                    with_ips += 1

                rows.append(result)

        print(
            f"Parsed {log_file.name} ({day}): "
            f"{file_count} alerts"
        )

    df = pd.DataFrame(rows)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n=== PARSE REPORT ===")
    print(f"Total lines:    {total_lines}")
    print(f"Alerts parsed:  {matched}")
    if matched > 0:
        print(
            f"With IP data:   {with_ips} "
            f"({with_ips/matched*100:.1f}%)"
        )
    print(f"\nSaved to: {output_path}")
    return str(output_path)

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / "data" / "alerts_monday_to_friday"
    OUT_DIR = ROOT / "outputs" / "snort"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_file = OUT_DIR / "snort_alerts.csv"
    result = parse_snort_alerts(str(DATA_DIR), str(out_file))