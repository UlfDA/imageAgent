# utils/config_handler.py

import configparser
import os

class ConfigHandler:
    def __init__(self, ini_path='config/init.ini'):
        self.ini_path = ini_path
        self.config = configparser.ConfigParser()
        if not os.path.exists(self.ini_path):
            self.config['IMAGE'] = {'path': ''}
            with open(self.ini_path, 'w') as configfile:
                self.config.write(configfile)
        else:
            self.config.read(self.ini_path)

    def save_image_path(self, path):
        self.config['IMAGE']['path'] = path
        with open(self.ini_path, 'w') as configfile:
            self.config.write(configfile)

    def get_image_path(self):
        return self.config['IMAGE'].get('path', '')
