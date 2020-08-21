import numpy as np

class Shape():

  def __init__(self):
    self.res = 50
    self.generateField(0, 50, self.res)

  def generateField(self, start, end, steps):

    '''Generates the a 2D signed distance field for a shape
    
            Parameters:
                    start (int): Start value of coordinates
                    end (int): End value of coordinates
                    steps (int): Number of steps in each dimension

            Returns:
                    outputField (list): List of signed distance field values
    '''
    self.start = start
    self.end = end
    self.steps = steps
    self.pts = np.float_([[x, y] 
                    for x in  np.linspace(start, end, steps) 
                    for y in np.linspace(start, end, steps)])

    self.field = np.float_(list(map(self.sdf, self.pts))).reshape(steps*steps, 1)


    # self.normalizeField()

    return True

  def normalizeField(self):

    '''Normalizes the signed distance field to be within [-1,1]'''

    absMin = abs(np.min(self.field))
    absMax = abs(np.max(self.field))

    absAbsMax = max(absMin, absMax)

    self.field /= absAbsMax
      

#subclasses

class Circle(Shape):

  def __init__(self, center : np.array, radius : float):


    self.center = center
    self.radius = radius / 2
    super().__init__()

  def sdf(self, p):

    return np.linalg.norm(p - self.center) - self.radius

class Box(Shape):

  def __init__(self, height: float, width: float, center: np.array):

    if(height <= 0 or width <= 0):
      raise ValueError("Height or Width cannot be negative")

    self.hw = np.float_((height / 2.0, width / 2.0))
    self.center = center
    super().__init__()

  def sdf(self, p):

    #translation
    p = p - self.center
    
    d = abs(p) - self.hw

    return np.linalg.norm([max(0, item) for item in d]) + min(max(d[0], d[1]), 0)
