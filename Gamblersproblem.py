import numpy as np
import sys
import matplotlib.pyplot as plt
if "../" not in sys.path:
  sys.path.append("../")

def value_iteration_for_gamblers(p_h, theta=0.0001, discount_factor=1.0):
    """
    Args:
        p_h: Probability of the coin coming up heads
    """
    
    V = np.zeros(101)
    policy = np.zeros([100, 99])
    rewards = np.zeros(101)
    rewards[100] = 1
    

    while True:
        delta = 0
        for s in range(1,100):
            v = V[s]
            action_vals = []
            for a in range(min(s, 100-s)+1):
                action_vals.append(p_h*(rewards[s+a]+discount_factor*V[s+a])+(1-p_h)*(rewards[s-a]+discount_factor*V[s-a]))
            action_vals = np.array(action_vals)
            best_action = np.argmax(action_vals) #np.random.choice(np.flatnonzero(action_vals == action_vals.max())) np.argmax(action_vals)
            policy[s] = np.eye(99)[best_action]
            V[s] = action_vals[best_action]
            delta = max(delta,abs(v-V[s]))
        if delta < theta:
            break
    
    return policy, V


policy, v = value_iteration_for_gamblers(0.55)

print([np.argmax(pol) for pol in policy])

print("Optimized Policy:")
print(policy)
print("")

print("Optimized Value Function:")
print(v)
print("")



# Plotting Final Policy (action stake) vs State (Capital)

# x axis values
x = range(100)
# corresponding y axis values
y = v[:100]
 
# plotting the points 
plt.plot(x, y)
 
# naming the x axis
plt.xlabel('Capital')
# naming the y axis
plt.ylabel('Value Estimates')
 
# giving a title to the graph
plt.title('Final Policy (action stake) vs State (Capital)')
 
# function to show the plot
plt.show()

# Plotting Final Policy (action stake) vs State (Capital)


# Plotting Capital vs Final Policy
# Plotting Capital vs Final Policy

# x axis values
x = range(100)
# corresponding y axis values
y = [np.argmax(pol) for pol in policy]
 
# plotting the bars
plt.bar(x, y, align='center', alpha=0.5)
 
# naming the x axis
plt.xlabel('Capital')
# naming the y axis
plt.ylabel('Final policy (stake)')
 
# giving a title to the graph
plt.title('Capital vs Final Policy')
 
# function to show the plot
plt.show()
# Implement!