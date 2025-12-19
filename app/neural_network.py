from activations import r, r_prime
import logging

logger = logging.getLogger(__name__)

class NeuralNetwork:
    def __init__(self, w1, w2, w3, w4, w5, w6, w7, w8, w9, b1, b2, b3, b4, eta):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        self.w6 = w6
        self.w7 = w7
        self.w8 = w8
        self.w9 = w9
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.b4 = b4
        self.eta = eta

    def calculate_a(self, first_weight, first_x, second_weight, second_x, bias):
        result = (first_weight * first_x) + (second_weight * second_x) + bias
        return r(result)
    
    def calculate_y_hat(self, a1, a2, a3):
        result = (self.w7 * a1) + (self.w8 * a2) + (self.w9 * a3) + self.b4
        logger.debug(f"y_hat: {result}")
        return result 

    def calculate_y(self, x1, x2):
        result = x1 + (2 * x2) - 1
        logger.debug(f"y: {result}")
        return result

    def calculate_dc_dy_hat(self):
        result = 2 * (self.y_hat - self.y)
        logger.debug(f"dc_dy_hat: {result}")
        return result
    
    def calculate_c(self, y_hat, y):
        result = (y_hat - y)**2
        logger.debug(f"L (c): {result}") 
        return result 
    
    def dc_dw_second_layer(self, dc_dy_hat, a, label):
        result = dc_dy_hat * a
        logger.debug(f"{label}: {result}")
        return result

    def dc_dw_first_layer(self, dc_dy_hat, dy_hat_da, first_weight, first_x, second_weight, second_x, bias, third_x, label):
        result = dc_dy_hat  * dy_hat_da * r_prime((first_weight * first_x) + (second_weight * second_x) + bias) * third_x
        logger.debug(f"{label}: {result}")
        return result 

    def update_weight(self, j, dc_dj, label, eta=0.01):
        result = j - eta * dc_dj
        logger.debug(f"{label}: {result}") 
        return result
    
    def forward_pass(self, x1, x2):
        logger.info(f"Performing a forward pass with x1: {x1} and x2: {x2}")
        self.a1 = self.calculate_a(self.w1, first_x=x1, second_weight=self.w4, second_x=x2, bias=self.b1)
        self.a2 = self.calculate_a(self.w2, first_x=x1, second_weight=self.w5, second_x=x2, bias=self.b2)
        self.a3 = self.calculate_a(self.w3, first_x=x1, second_weight=self.w6, second_x=x2, bias=self.b3)
        logger.debug(f"a1: {self.a1}, a2: {self.a2}, a3: {self.a3}")
        
        self.y_hat = self.calculate_y_hat(self.a1, self.a2, self.a3)
        self.y = self.calculate_y(x1, x2)
        c = self.calculate_c(self.y_hat, self.y)
        return c
    
    def back_propagation(self, x1, x2):
        logger.info(f"Performing back propagation with x1: {x1} and x2: {x2}")
        dc_dy_hat = self.calculate_dc_dy_hat() 

        parameter_sets = [
            (self.w7, self.w1, x1, self.w4, x2, self.b1, x1, 'dc_w1'),
            (self.w8, self.w2, x1, self.w5, x2, self.b2, x1, 'dc_w2'), 
            (self.w9, self.w3, x1, self.w6, x2, self.b3, x1, 'dc_w3'),
            (self.w7, self.w1, x1, self.w4, x2, self.b1, x2, 'dc_w4'),
            (self.w8, self.w2, x1, self.w5, x2, self.b2, x2, 'dc_w5'), 
            (self.w9, self.w3, x1, self.w6, x2, self.b3, x2, 'dc_w6'),
            (self.w7, self.w1, x1, self.w4, x2, self.b1, 1, 'dc_b1'),
            (self.w8, self.w2, x1, self.w5, x2, self.b2, 1, 'dc_b2'),
            (self.w9, self.w3, x1, self.w6, x2, self.b3, 1, 'dc_b3')
        ]
        dc_w1, dc_w2, dc_w3, dc_w4, dc_w5, dc_w6, dc_b1, dc_b2, dc_b3 = [
            self.dc_dw_first_layer(dc_dy_hat, *param)
                for param in parameter_sets
        ]
        dc_b4 = dc_dy_hat

        # calculate second layer partial derivatives for weights w7-w9
        dc_w7 = self.dc_dw_second_layer(dc_dy_hat, self.a1, 'dc_w7')
        dc_w8 = self.dc_dw_second_layer(dc_dy_hat, self.a2, 'dc_w8')
        dc_w9 = self.dc_dw_second_layer(dc_dy_hat, self.a3, 'dc_w9')

        self.w1 = self.update_weight(self.w1, dc_w1, 'w1')
        self.w2 = self.update_weight(self.w2, dc_w2, 'w2')
        self.w3 = self.update_weight(self.w3, dc_w3, 'w3')
        self.w4 = self.update_weight(self.w4, dc_w4, 'w4')
        self.w5 = self.update_weight(self.w5, dc_w5, 'w5')
        self.w6 = self.update_weight(self.w6, dc_w6, 'w6')
        self.w7 = self.update_weight(self.w7, dc_w7, 'w7')
        self.w8 = self.update_weight(self.w8, dc_w8, 'w8')
        self.w9 = self.update_weight(self.w9, dc_w9, 'w9')

        self.b1 = self.update_weight(self.b1, dc_b1, 'b1')
        self.b2 = self.update_weight(self.b2, dc_b2, 'b2')
        self.b3 = self.update_weight(self.b3, dc_b3, 'b3')
        self.b4 = self.update_weight(self.b4, dc_b4, 'b4')

    def train(self, x1_list, x2_list, epochs):
        logger.info("Starting training...")
        loss = None
        for e in range(epochs):
            for x1, x2 in zip(x1_list, x2_list):
                loss = self.forward_pass(x1, x2)
                logger.info(f"Epoch: {e}, loss: {loss}")
                self.back_propagation(x1, x2)
        return loss