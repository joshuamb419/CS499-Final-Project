from minigrid.core.world_object import Wall

def step(env, action):
 
        x, y = env.agent_pos
        
        if action == 0: #Check if action is valid then set the new position if true
            env.agent_dir = 0
            if x + 1 < env.width: #Not border
                tile = env.grid.get(x + 1, y)
                if tile is None or tile.type != "wall": #Not wall
                    env.agent_pos = (x + 1, y)
                    
        elif action == 1:
            env.agent_dir = 1
            if y + 1 < env.height:
                tile = env.grid.get(x, y + 1)
                if tile is None or tile.type != "wall":
                    env.agent_pos = (x, y + 1)
                    
        elif action == 2:
            env.agent_dir = 2
            if x - 1 >= 0:
                tile = env.grid.get(x - 1, y)
                if tile is None or tile.type != "wall":
                    env.agent_pos = (x - 1, y)
                    
        elif action == 3:
            env.agent_dir = 3
            if y - 1 >= 0:
                tile = env.grid.get(x, y - 1)
                if tile is None or tile.type != "wall":
                        env.agent_pos = (x, y - 1)
            
        if env.agent_pos == env.goal_pos: #Check if goal
            
            reward = 100.0  
            finished = True
            
        else: #Deduct if not

            reward = -0.5
            finished = False
            
        env.step_count += 1
        if env.step_count >= env.max_steps:
            
            time_out = True
            
        else:
            
            time_out = False

            
        return reward, finished, time_out