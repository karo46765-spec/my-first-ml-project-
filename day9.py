import numpy as np

dataset = np.array(
    [
        [1, 720],
        [2, 580],
        [3, 640],
        [4, 680],
        [5, 790],
        [6, 610],
        [7, 750],
        [8, 700],
        [9, 540],
        [10, 820],
    ]
)
np.random.seed(40)
shuffled_dataset = dataset.copy()
np.random.shuffle(shuffled_dataset)
print("---Shuffeld Dataset____")
print(shuffled_dataset)
total_rows = len(shuffled_dataset)

Split_index = int(total_rows * 0.8)

eighty_percent = shuffled_dataset[:Split_index]
twenty_percent = shuffled_dataset[Split_index:]

# Split the array automatically at the 80% mark
eighty_percent2, twenty_percent2 = np.split(shuffled_dataset, [Split_index])

from sklearn.model_selection import train_test_split

# test_size=0.2 automatically allocates 20% to the second variable
train_data3, test_data3 = train_test_split(dataset, test_size=0.2, random_state=42)

print("All splits training and testing")
print(f"(80%-{(train_data3)}):")
print(f"(20%-{(test_data3)}):")
print(f"(80%-{(eighty_percent)}):")
print(f"(20%-{(twenty_percent)}):")
print(f"(80%-{(eighty_percent2)}):")
print(f"(20%-{(twenty_percent2)}):")
