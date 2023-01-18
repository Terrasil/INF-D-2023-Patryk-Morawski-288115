import pygad
import numpy as np
import imageio as iio
from difflib import SequenceMatcher
import time
import threading

""" Definiowanie zmiennych """

def set_nonfilled_color(color_palette, color):
    _color = np.array(color, dtype=np.uint8)
    
    if _color.shape[0] > color_palette.shape[1]:
        for n in range(_color.shape[0] - color_palette.shape[1]):
            _color = np.delete(_color, _color.shape[0] - 1)
    else:
        for n in range(color_palette.shape[1] - _color.shape[0]):
            _color = np.append(_color, [255])
            
    new_color_palette = [_color]
    for color_in_palette in color_palette:
        if not (color_in_palette == _color).all():
            new_color_palette = np.vstack([new_color_palette, color_in_palette])
    return new_color_palette

def find_color(color_palette, color):
    index = 0
    for color_in_palette in color_palette:
        if np.array_equal(color_in_palette, color):
            return index
        index += 1
    return None

def use_palette_on_img(color_palette,_img):
    tmp_img = np.full((_img.shape[0], _img.shape[1]), 0)
    for col in range(_img.shape[0]):
        for row in range(_img.shape[1]):
            tmp_img[col][row] = find_color(color_palette, _img[col][row])
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
print("");

def fitness_function_factory(img, nonogram_solution):
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
    return fitness

""" Test """
def run_test(title, img_path, nonfilled_color, count):
    
    print("Uruchomienie testu: " + title)
    """ Inizjalizacja danych """
    raw_img = np.array(iio.imread(img_path))
    pixels = np.reshape(raw_img, (raw_img.shape[0] * raw_img.shape[1], raw_img.shape[2]))
    color_palette = np.unique(pixels, axis=0)
    color_palette = set_nonfilled_color(color_palette, nonfilled_color)    
    img = use_palette_on_img(color_palette, raw_img)
    nonogram_solution = nonogram_solution_build(img)
    
    """ Parametry chromosomu """
    gene_space = [*range(len(color_palette))]
    num_genes = img.shape[0] * img.shape[1]
    
    successfull = 0
    total_time = 0
    min_time = None
    max_time = None
    total_gen = 0
    max_gen = None
    min_gen = None
    total_similarity = 0
    max_similarity = None
    min_similarity = None
    for i in range(count):
        ta = time.time()
        
        ga_instance = pygad.GA(gene_space=gene_space,
                                num_generations=num_generations,
                                num_parents_mating=num_parents_mating,
                                fitness_func=fitness_function_factory(img, nonogram_solution),
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
            
            total_gen += ga_instance.generations_completed
            
            if max_time == None or tb-ta > max_time:
                max_time = tb-ta
            if min_time == None or tb-ta < min_time:
                min_time = tb-ta
            
            if max_gen == None or ga_instance.generations_completed > max_gen:
                max_gen = ga_instance.generations_completed
            if min_gen == None or ga_instance.generations_completed < min_gen:
                min_gen = ga_instance.generations_completed
        
        total_similarity += 1 - np.mean(solution != img.flatten())
            
        if max_similarity == None or 1 - np.mean(solution != img.flatten()) > max_similarity:
            max_similarity = 1 - np.mean(solution != img.flatten())
        if min_similarity == None or 1 - np.mean(solution != img.flatten()) < min_similarity:
            min_similarity = 1 - np.mean(solution != img.flatten())
    
    print("\nWyniki dla: " + title + ": ")
    print("Successfull:", successfull/count if successfull > 0 else 0.0, "("+str(successfull)+"/"+str(count)+")");
    print("Min time:", min_time, "sec");
    print("Max time:", max_time, "sec");
    print("Avg time:", total_time/successfull if successfull > 0 else None, "sec");
    print("Min generations:", min_gen);
    print("Max generations:", max_gen);
    print("Avg generations:", round(total_gen/successfull) if successfull > 0 else None);
    print("Min similarity:", min_similarity);
    print("Max similarity:", max_similarity);
    print("Avg similarity:", total_similarity/count if total_similarity > 0 else 0.0);
    
    

""" Testy i statystyka """

threads = []
threads.append(threading.Thread(target=run_test, args=("Test dla monogramu 7x7 2 kolory", "pixelarts/7x7-2.png", [255, 255, 255], 10)))
threads.append(threading.Thread(target=run_test, args=("Test dla monogramu 4x4 2 kolory", "pixelarts/4x4-2.png", [255, 255, 255], 10)))
threads.append(threading.Thread(target=run_test, args=("Test dla monogramu 3x3 2 kolory", "pixelarts/3x3-2.png", [255, 255, 255], 10)))
threads.append(threading.Thread(target=run_test, args=("Test dla monogramu 2x2 2 kolory", "pixelarts/2x2-2.png", [255, 255, 255], 10)))

for thread in threads:
    thread.start()
    
for thread in threads:
    thread.join()