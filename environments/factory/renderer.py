import numpy as np
import pygame
from pathlib import Path


class Renderer:
    BG_COLOR = (99, 110, 114)
    WHITE = (200, 200, 200)

    def __init__(self, grid_w=16, grid_h=16, cell_size=25, fps=4, grid_lines=True, assets=['wall', 'agent']):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.cell_size = cell_size
        self.fps = fps#
        self.grid_lines = grid_lines
        pygame.init()
        self.screen_size = (grid_h*cell_size, grid_w*cell_size)
        self.screen = pygame.display.set_mode(self.screen_size)
        self.clock = pygame.time.Clock()
        self.assets = {name: self.load_asset(name, 0.97) for name in assets}
        self.fill_bg()

    def fill_bg(self):
        self.screen.fill(Renderer.BG_COLOR)
        if self.grid_lines:
            h, w = self.screen_size
            for x in range(0, w, self.cell_size):
                for y in range(0, h, self.cell_size):
                    rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, Renderer.WHITE, rect, 1)

    def render_asset(self, r, c, name):
        img = self.assets[name]
        o = self.cell_size//2
        r_, c_ = r*self.cell_size + o, c*self.cell_size + o
        rect = img.get_rect()
        rect.centerx, rect.centery = c_, r_
        self.screen.blit(img, rect)
        return c_, r_

    def load_asset(self, name, factor=1.0):
        s = int(factor*self.cell_size)
        wall_img = pygame.image.load(str(Path(__file__).parent / 'assets' / f'{name}.png')).convert_alpha()
        wall_img = pygame.transform.scale(wall_img, (s, s))
        return wall_img

    def render(self, pos_dict):
        self.fill_bg()
        for asset, positions in pos_dict.items():
            for x, y in positions:
                self.render_asset(x, y, asset)
        pygame.display.flip()
        self.clock.tick(self.fps)


if __name__ == '__main__':
    renderer = Renderer(fps=2, cell_size=40, assets=['wall', 'agent', 'dirt'])
    for i in range(15):
        renderer.render({'agent': [(5, 5)], 'wall': [(0, i), (i, 0)], 'dirt': [(3,3), (3,4)]})
