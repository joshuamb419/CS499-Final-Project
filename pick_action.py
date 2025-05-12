import random

def pick_action(state, values, eps):
        
        best_action = -1
        best_value = -10000
        x, y = state
        
        if random.random() < eps:
            
            eps_action = random.choice([0, 1, 2, 3]) #Choose a random action if less than eps
            
            return eps_action
        
        above_state = (x, y - 1)
        right_state = (x + 1, y)
        below_state = (x, y + 1)
        left_state = (x - 1, y)
        
        if values[above_state] > best_value: #Check all actions and keep the highest value
            
            best_value = values[above_state]
            best_action = 3
            
        elif values[below_state] > best_value:
            
            best_value = values[below_state]
            best_action = 1
            
        elif values[right_state] > best_value:
            
            best_value = values[right_state]
            best_action = 0
            
        elif values[left_state] > best_value:
            
            best_value = values[left_state]
            best_action = 2
            
        return best_action