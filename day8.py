import numpy as np
applications = np.array([[0.9,0.2],[0.4,0.8],[0.6,0.4]])
weights = np.array([0.7 , -0.3])

print("Application Shape" , applications.shape)
print("weight shape" , weights.shape)

final_score = np.dot(applications,weights)

python_finalscore = applications @ weights

print(final_score)
print(python_finalscore)


for index, score in enumerate(final_score):
    print(f"Applications : {index+1}, Scores : {score}")
