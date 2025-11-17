from flappy_bird_env import FlappyBirdEnv


class FlappyBirdEnvWithImageObs(FlappyBirdEnv):
    
    @property
    def reward(self):
        if any([not pipe.passed and pipe.x < self._bird.x
                for pipe in self._pipes]):
            return 1
        elif not self.terminated:
            return 0.01
        else:
            return -1