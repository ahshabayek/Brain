- ![image.png](../assets/image_1720590080599_0.png)
- ![image.png](../assets/image_1720590487358_0.png)
- ![image.png](../assets/image_1720590724018_0.png)
- ![image.png](../assets/image_1722118995812_0.png)
-
- ![image.png](../assets/image_1720591990657_0.png)
-
- **Lesson 21 - Configuration Design in Knowledge-Based AI**
  
  Configuration design is a method used in knowledge-based AI to solve configuration problems by starting with specific constraints or specifications and generating a detailed arrangement model of all the components involved. This approach is particularly useful for tasks where the components are already known, and the goal is to find the optimal arrangement of these components to meet certain constraints.
- ### Key Concepts
- **Specifications and Constraints**:
	- The process begins with a set of specifications, such as constraints on dimensions, materials, costs, and other relevant parameters.
	- An example provided is configuring David’s basement with specific constraints.
- **Abstract and Partial Solutions**:
	- The initial solutions may be abstract or partial, represented in the form of a design plan.
	- Each plan specifies a subset of variables, which are gradually assigned values to refine and complete the plan iteratively.
- **Arrangement Model**:
	- The final output is an arrangement model, detailing the placement and configuration of all components.
	- For instance, the process may involve detailing the layout of a house, breaking down into plans for each floor, then for specific areas like kitchens and bedrooms, refining further to specifics within each area.
- **Two-Way Arrows**:
	- There is a feedback loop between the specifications, configuration process, and the arrangement model.
	- This allows for reassessment and adjustment of specifications based on the evolving solution, highlighting the co-evolution of problem and solution.
- ### Example: Configuring a Chair
  
  To illustrate configuration design in detail, consider the task of configuring a chair:
- **Knowledge Representation**:
	- All knowledge about the chair is represented in a frame structure, with slots for various attributes like size, material, and cost.
	- Subcomponents like legs, seat, and back are also represented with their specific attributes.
- **Range of Values**:
	- Each variable can take a range of values, e.g., the seat can weigh between 10 to 100 grams.
	- Costs are determined based on material and size, with specific values for each attribute.
- **Applying Constraints**:
	- Given a specification (e.g., chair weighs over 200 grams, costs at most $20, and has four legs), the configuration process fills in values for each variable to meet these constraints.
	- An abstract plan might first allocate costs, distribute it evenly among components, and refine details like material and weight.
- ### Exercise Example
  
  A practical exercise involves configuring a chair with a specified maximum cost and a metal seat of a specific weight:
- **Step-by-Step Process**:
	- Initial constraints are noted, and heuristic rules are applied to minimize costs while meeting the specifications.
	- Variables like the number of legs, material for each part, and their costs are determined iteratively.
	- The final solution ensures all constraints are satisfied.
- ### Connection to Other Concepts
- **Classification**:
	- Configuration uses classification’s notion of prototypical concepts but focuses on creating arrangements rather than just categorizing.
	- It involves extending plans, assigning values, refining, and expanding.
- **Case-Based Reasoning**:
	- Both methods address routine design problems but start differently.
	- Configuration starts with a prototype and assigns variable values, while case-based reasoning tweaks a specific past design to meet current constraints.
- **Planning**:
	- Configuration can be seen as a specialized form of planning where skeletal plans are instantiated and refined.
- ### Assignment
  
  The lesson concludes with an assignment to design an agent for solving Raven’s Progressive Matrices using the principles of configuration:
- **Leverage Constraint Propagation**:
	- Identify variables and their possible values.
	- Utilize old plans and adapt them to new problems.
- ### Wrap-Up
  
  Configuration is a crucial routine design task in knowledge-based AI, involving detailed planning and iterative refinement to meet given constraints. The knowledge of prototypical concepts, heuristic rules, and the ability to reassess and adjust plans are fundamental to this process.