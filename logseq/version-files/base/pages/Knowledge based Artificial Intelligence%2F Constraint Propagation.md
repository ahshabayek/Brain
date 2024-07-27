- ![image.png](../assets/image_1720553718272_0.png)
-
- ![image.png](../assets/image_1720555762391_0.png)
- ![image.png](../assets/image_1720555923931_0.png)
- # Lesson 20 - Constraint Propagation
- ### Introduction to Constraint Propagation
  
  Constraint propagation is a method of inference that assigns values to variables characterizing a problem in such a way that some conditions are satisfied. Given a problem, it is characterized by a set of variables. The task is to assign specific values to each variable so that global constraints are satisfied. For example, let’s consider a figure drawn on a 2D surface. The problem is to determine if it represents a 3D object. Here, the variables are surfaces and orientations.
- ### Example of Constraint Propagation
  
  Imagine a 2D figure that can be seen as a 3D object with four surfaces, each having a specific orientation. The method of constraint propagation helps identify the surfaces and their orientations. The constraints are defined by junctions where three surfaces meet, each having particular properties. Assignments of surfaces and orientations must satisfy these constraints. Sometimes, multiple interpretations can simultaneously satisfy all constraints, leading to ambiguity. If no assignment satisfies all constraints, interpretation becomes difficult.
- ### Constraints in Language Processing
  
  Consider the sentence "Colorless green ideas sleep furiously." It is semantically meaningless but grammatically correct. The variables are lexical categories like words, nouns, and predicates. The constraints are defined by grammar rules. Assignments must satisfy these grammatical constraints. This process helps determine the grammatical correctness of a sentence.
- ### Decomposition in Constraint Propagation
  
  In computer vision, constraint propagation can be used to recognize 3D objects from 2D images. For instance, recognizing a cube involves detecting edges and lines, grouping these lines into surfaces, and then recognizing the 3D object. This process involves decomposing the task into subtasks, such as detecting edges, grouping lines into surfaces, and finally recognizing the 3D object.
- ### Types of Junctions in Trihedral Objects
  
  Different types of junctions can be identified in trihedral objects like cubes:
- **Y Junction:** Each line represents a fold where two surfaces meet.
- **L Junction:** Characterized by two blades.
- **W Junction:** Consists of blades and folds.
- ### Exercise: Identifying Junctions
  
  Let's practice identifying junctions in a cube. For each junction, determine the type (L, Y, W, or T) and identify the edges as fold or blade.
- ### Application to Complex Images
  
  In more complex images, a more detailed ontology of constraints is required. For example, a junction could be characterized by different combinations of folds and blades. Additional conventions, such as labeling edges next to the background as blades, can help resolve ambiguities.
- ### Constraint Propagation in Natural Language Processing
  
  Returning to the sentence "Colorless green ideas sleep furiously," we can use a simple grammar to assign values to words, ensuring the sentence is grammatically correct. A sentence can be divided into a noun phrase and a verb phrase. Each word is assigned a lexical category based on the grammar rules, satisfying the constraints.
- ### Assignment: Constraint Propagation in Raven’s Progressive Matrices
  
  For the final project, think about how constraint propagation can be applied to reason over images directly. Identify primitive constraints and propagate them to understand the image. Decide whether to abstract out propositional representations or stick to visual reasoning.
- ### Wrap Up
  
  Today, we discussed constraint propagation, a method of inference where values are assigned to variables to satisfy constraints. This process helps interpret images and understand language by using prior knowledge of constraints. While constraint propagation is complex, it’s a powerful tool in both visual and language processing. We’ll explore this further in future lessons on visual and spatial reasoning.
- ### The Cognitive Connection
  
  Constraint propagation is a general-purpose method in both AI and human cognition. It allows us to use knowledge of the world to make sense of it. Constraints can be symbolic or numeric. For example, in spreadsheets, changes in one column propagate through formulas to other columns, illustrating numerical constraint propagation.
  
  ---
  
  This structured approach to understanding constraint propagation helps break down the concept into manageable sections, facilitating better comprehension and application in various fields.v