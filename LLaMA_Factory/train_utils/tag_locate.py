

def find_occurrence(tensor, template):

    if template == "llama3":
        print("llama3")
        target_value=128007
        indices = []
        for row in tensor:
            occurrences = (row == target_value).nonzero(as_tuple=True)[0]
            if len(occurrences) >= 2:
                indices.append(occurrences[1].item() + 1)
            else:
                indices.append(-1)  

    elif template == "llama2":
        print("llama2")
        target_value=25580
        indices = []
        for row in tensor:
            occurrences = (row == target_value).nonzero(as_tuple=True)[0]
            if len(occurrences) >= 2:
                indices.append(occurrences[1].item() + 1)
            else:
                indices.append(-1)  

    elif template == "mistral":
        print("mistral")
        target_value=4
        indices = []
        for row in tensor:
            occurrences = (row == target_value).nonzero(as_tuple=True)[0]
            if len(occurrences) > 0:
                indices.append(occurrences[0].item()-1)
            else:
                indices.append(-1) 

    elif template == "qwen2":
        print("qwen")
        target_value=151644
        indices = []
        for row in tensor:
            occurrences = (row == target_value).nonzero(as_tuple=True)[0]
            if len(occurrences) >= 3:
                indices.append(occurrences[2].item() + 1)
            else:
                indices.append(-1) 
    else:
        print("Unknown Template!")
        return None

    return indices