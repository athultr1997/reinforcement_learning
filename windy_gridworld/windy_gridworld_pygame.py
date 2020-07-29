import pygame
import numpy as np

# defining colours
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Defining the grid cell properties
WIDTH = 30
HEIGHT = 30
MARGIN = 5

# creating a two-dimensional array
grid = np.zeros((7,10))
grid[0,5] = 1

# initialize pygame modules
pygame.init()

# defining the screen size
SCREEN_WIDTH = 355
SCREEN_HEIGHT = 250

# defining the screen object
screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])

# setting the caption for the screen
pygame.display.set_caption('Windy Gridworld')

# loop until the user clicks the close button
done = False

# defining time control for the screen update rate
clock = pygame.time.Clock()

# defining the text to be displayed at the start and goal positions
font = pygame.font.SysFont('Calibri', 40, True, False)
start_text = font.render('S', True, BLUE)
start_text_rect = start_text.get_rect(center = [(MARGIN + WIDTH) * 0 + MARGIN + (WIDTH / 2), (MARGIN + HEIGHT) * 3 + MARGIN + (HEIGHT / 2)])
goal_text = font.render('G', True, RED)
goal_text_rect = goal_text.get_rect(center = [(MARGIN + WIDTH) * 7 + MARGIN + (WIDTH / 2), (MARGIN + HEIGHT) * 3 + MARGIN + (HEIGHT / 2)])

while not done:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			done = True
		elif event.type == pygame.MOUSEBUTTONDOWN:
			pos = pygame.mouse.get_pos()
			column = pos[0]//(WIDTH + MARGIN)
			row = pos[1]//(HEIGHT + MARGIN)
			grid[row,column] = 1
			print('Click ', pos, '; Grid coordinates: ', row,',',column)

	screen.fill(BLACK)

	for row in range(grid.shape[0]):
		for column in range(grid.shape[1]):
			color = WHITE
			if grid[row, column] == 1:
				color = GREEN
			pygame.draw.rect(screen, color, [(MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN, WIDTH, HEIGHT])

	screen.blit(start_text, start_text_rect)
	screen.blit(goal_text, goal_text_rect)
	clock.tick(60)

	pygame.display.flip()

pygame.quit()