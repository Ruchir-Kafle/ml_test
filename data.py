# Dataset
data: list[list[str | int]] = [
    ["Alice", 133, 65, "F"],
    ["Bob", 160, 72, "M"],
    ["Charlie", 152, 70, "M"],
    ["Diana", 120, 60, "F"]
]


def normalize_dataset(dataset: list[list[str | int]]) -> list[list[str | int]]:
    final_data: list[list[str | int]] = []
    mean_weight: int = 0
    mean_height: int = 0

    for row in dataset:
        mean_weight += row[1]
        mean_height += row[2]
    mean_weight /= len(dataset)
    mean_height /= len(dataset)

    for row in dataset:
        new_row: list[str | int] = []
        # new_row.append(row[0])
        
        # Actual weight and height
        # new_row.append(row[1] - mean_weight)
        # new_row.append(row[2] - mean_height)

        # Weight and height to match article
        new_row.append(row[1] - 135)
        new_row.append(row[2] - 66)

        if row[3] == "F": new_row.append(1)
        elif row[3] == "M": new_row.append(0)

        final_data.append(new_row)

    return final_data