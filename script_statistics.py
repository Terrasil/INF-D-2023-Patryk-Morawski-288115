import pygad
import numpy as np
import imageio as iio
from difflib import SequenceMatcher
import time

""" Definiowanie zmiennych """
raw_img = []
pixels = []
color_palette = []
nonogram_solution = []
img = []
nonogram_solution = []

def set_nonfilled_color(color):
    _color = np.array(color, dtype=np.uint8)
    new_color_palette = [_color]
    for color_in_palette in color_palette:
        if not (color_in_palette == _color).all():
            new_color_palette = np.vstack([new_color_palette, color_in_palette])
    return new_color_palette

def find_color(color):
    index = 0
    for color_in_palette in color_palette:
        if np.array_equal(color_in_palette, color):
            return index
        index += 1
    return None

def use_palette_on_img(_img):
    tmp_img = np.full((_img.shape[0], _img.shape[1]), 0)
    for col in range(_img.shape[0]):
        for row in range(_img.shape[1]):
            tmp_img[col][row] = find_color(_img[col][row])
    return tmp_img

""" Rozwiązanie monogramu """
def nonogram_solution_build(_img):
    answer = [[],[]]
    
    # Wiersze
    for col in range(_img.shape[0]):
        recipe = []
        color = 0
        count = 0
        for row in range(_img.shape[1]):
            if _img[col][row] != 0:
                if _img[col][row] == color:
                    count += 1
                else:
                    if color != 0 and count != 0:
                        recipe.append([color, count])
                    color = _img[col][row]
                    count = 1
            else:
                if color != 0 and count != 0:
                    recipe.append([color, count])
                color = 0
                count = 0
                
            if row == _img.shape[1] - 1:
                if color != 0 and count != 0:
                    recipe.append([color, count])
        answer[0].append(recipe) 
                
    # Kolumny
    for col in range(_img.shape[1]):
        recipe = []
        color = 0
        count = 0
        for row in range(_img.shape[0]):
            if _img[row][col] != 0:
                if _img[row][col] == color:
                    count += 1
                else:
                    if color != 0 and count != 0:
                        recipe.append([color, count])
                    color = _img[row][col]
                    count = 1
            else:
                if color != 0 and count != 0:
                    recipe.append([color, count])
                color = 0
                count = 0
                
            if row == _img.shape[1] - 1:
                if color != 0 and count != 0:
                    recipe.append([color, count])
        answer[1].append(recipe) 
                
    return answer
        
""" Parametry populacji """
sol_per_pop = 100
num_generations = 100

""" Parametry mutacji """
num_parents_mating = 5
keep_parents = 2
crossover_type = "scattered"
mutation_type = "random"
mutation_percent_genes = 1
parent_selection_type = 'rank'

""" Parametry metody fitness """
acceptable_threshold = 1.0
calc_fitness_on_similarity_cond = False

print("Instance parameters:");
print("sol_per_pop:", sol_per_pop);
print("num_generations:", num_generations);
print("num_parents_mating:", num_parents_mating);
print("keep_parents:", keep_parents);
print("crossover_type:", crossover_type);
print("mutation_type:", mutation_type);
print("mutation_percent_genes:", mutation_percent_genes);
print("parent_selection_type:", parent_selection_type);
print("acceptable_threshold:", acceptable_threshold);
print("calc_fitness_on_similarity_cond:", calc_fitness_on_similarity_cond);

def fitness(solution, solution_idx):
    fitness_solution_nonogram = nonogram_solution_build(
        solution.reshape((img.shape[0], img.shape[1])))
    fitness = 0
    d = 0
    for direction in fitness_solution_nonogram:
        l = 0
        for line in direction:
            line_flat = np.array(line).flatten()
            sol_flat = np.array(nonogram_solution[d][l]).flatten()
            similarity = SequenceMatcher(None, line_flat, sol_flat).ratio()
            if similarity >= acceptable_threshold:
                if calc_fitness_on_similarity_cond:
                    fitness += similarity
                else:
                    fitness += 1
            l += 1
        d += 1
    return -(img.shape[0] + img.shape[1]) + fitness;

""" Test """
def run_test(img_path, nonfilled_color, count):
    global raw_img
    global pixels
    global color_palette
    global img
    global nonogram_solution
    
    """ Inizjalizacja danych """
    raw_img = np.array(iio.imread(img_path))
    pixels = np.reshape(raw_img, (raw_img.shape[0] * raw_img.shape[1], raw_img.shape[2]))
    color_palette = np.unique(pixels, axis=0)
    color_palette = set_nonfilled_color(nonfilled_color)    
    img = use_palette_on_img(raw_img)
    nonogram_solution = nonogram_solution_build(img)
    
    """ Parametry chromosomu """
    gene_space = [*range(len(color_palette))]
    num_genes = img.shape[0] * img.shape[1]
    
    total_time = 0
    successfull = 0
    for i in range(count):
        ta = time.time()
        
        ga_instance = pygad.GA(gene_space=gene_space,
                                num_generations=num_generations,
                                num_parents_mating=num_parents_mating,
                                fitness_func=fitness,
                                sol_per_pop=sol_per_pop,
                                num_genes=num_genes,
                                parent_selection_type=parent_selection_type,
                                keep_parents=keep_parents,
                                crossover_type=crossover_type,
                                mutation_type=mutation_type,
                                mutation_percent_genes=mutation_percent_genes,
                                stop_criteria = ["saturate_100000", "reach_0"])
        ga_instance.run()
            
        tb = time.time()
        
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        
        if 1 - np.mean(solution != img.flatten()) == 1.0:
            successfull += 1
            total_time += tb-ta
            
    print("Successfull:", successfull/count if successfull > 0 else 0.0, "("+str(successfull)+"/"+str(count)+")");
    print("Avg time:", total_time/successfull if successfull > 0 else None, "sec");
    
    

""" Statystyka """ # Dodać zabezpieczenie obslugujące brak kanału alfa lub jego posiadanie
print("Test dla monogramu 7x7 2 kolory:")
run_test("pixelarts/7x7-2.png", [255, 255, 255], 50)
print("Test dla monogramu 4x4 2 kolory:")
run_test("pixelarts/4x4-2.png", [255, 255, 255], 50)
print("Test dla monogramu 3x3 2 kolory:")
run_test("pixelarts/3x3-2.png", [255, 255, 255, 255], 50)
print("Test dla monogramu 2x2 2 kolory:")
run_test("pixelarts/2x2-2.png", [255, 255, 255, 255], 50)