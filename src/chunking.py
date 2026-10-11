# fixed size chunking 

def fixed_size_chunk(text: str, chunk_size: int, chunk_overlap:int) -> list[str]:

    text = text.strip() 

    if not text: 
        return [] 

    step = chunk_size - chunk_overlap 

    chunks = [] 

    start = 0 

    while start < len(text): 

        chunk = text[start: start + chunk_size].strip() 

        if chunk:
            chunks.append(chunk)

        start += step 

    return chunks         