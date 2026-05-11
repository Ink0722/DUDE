def make_conversation(example):
    try:
        messages = example.get("messages", [])
        images = example.get("images", [])
        return {
            "prompt": messages,
            "image": images,
        }
    except Exception:
        return {
            "prompt": example.get("messages", []),
            "image": example.get("images", []),
        }


def format_url(image_path: str, local=False) -> str:
    if image_path.startswith("http://") or image_path.startswith("https://"):
        return image_path

    with open(image_path, "rb") as f:
        b = f.read()

    import base64

    b64 = base64.b64encode(b).decode("ascii")
    return f"data:image/png;base64,{b64}"


def add_row(sample, name, value):
    sample[name] = value
    return sample
