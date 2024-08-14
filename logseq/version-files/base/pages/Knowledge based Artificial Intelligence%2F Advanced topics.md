- visual spacial knowledge/ reasoning
-
- Humans
	- analogical presentations
	- propositional presentation(also computers)
- ![image.png](../assets/image_1722042788575_0.png)
- ![image.png](../assets/image_1722071200169_0.png)
- ![image.png](../assets/image_1722071251964_0.png)
- Certainly! Here's a detailed summary and highlights of the lecture on Visuospatial Reasoning without omitting any details:
  
  ---
- ### 608 - Visuospatial Reasoning Introduction
  
  **Visuospatial reasoning** involves reasoning with visuospatial knowledge, which has two components:
- **Visual**: Concerns the "what" (e.g., the sun).
- **Spatial**: Concerns the "where" (e.g., the sun is at the top right of the picture).
  
  Example: A picture with a sun at the top right. The sun is the "what," and the top right position is the "where."
  
  Visuospatial reasoning has been encountered in constraint propagation for line labeling in 2D images. Visuospatial knowledge often implies causality. For example, seeing a cup and a pool of water, one infers the water spilled from the cup.
- ### 609 - Two Views of Reasoning
  
  There are different ways to handle visuospatial knowledge:
- **Propositional Representations**: Extracting logical or rule-based representations from visual figures (e.g., a triangle with an apex facing right can be represented propositionally). AI agents can use these propositional representations for reasoning.
- **Analogical Representations**: Structural correspondence between the representation and the external figure (e.g., a triangle is represented as a triangle). These are close to perceptual modality and are used in human cognition.
  
  Computers often use propositional representations, but analogical representations align more with human cognition. This distinction also applies to other perceptual modalities, like auditory processing.
- ### 610 - Symbol Grounding Problem
  
  The lecture summarized the content and encoding of knowledge:
- **Content**: Visuospatial knowledge includes "what" (visual) and "where" (spatial).
- **Encoding**: Can be analogical (structural correspondence) or propositional.
  
  Example: A script for going to a restaurant can be represented propositionally (tracks, props, actors) or analogically (a short movie).
  
  The course often deals with propositional representations of verbal knowledge. Understanding visuospatial knowledge and analogical representation is crucial for developing AI that can process such knowledge.
- ### 611 - Visuospatial Reasoning: An Example
  
  **Galatea**, a system developed by Jim Davies at Georgia Tech, used visuospatial reasoning to solve the Duncker problem:
- **Duncker Problem Story**: An army decomposes into smaller groups to avoid mines and overthrow a king's fortress.
- **Duncker Problem for AI**: A physician must use a laser gun on a tumor without harming healthy tissue. The solution involves splitting the laser beam, analogous to splitting the army.
  
  Galatea solved the problem using visuospatial knowledge without extracting causal patterns explicitly. It transferred visuospatial knowledge from the original story to the new problem step-by-step.
- ### 612 - Visuospatial Reasoning: Another Example
  
  **Archytas**, an AI program developed by Patrick Yaner, extracted causal models from vector graphics drawings (e.g., a piston and crankshaft system):
- **Process**: Archytas used a library of known drawings with segmented shapes and behavioral models.
- **Function**: Given a new drawing, it matched segments to known shapes, creating a causal and functional model.
  
  Archytas demonstrated how AI could understand and extract stories from engineering diagrams using visuospatial knowledge and analogical reasoning.
- ### 613 - Ravens Progressive Matrices
  
  Recent advancements in AI for solving Raven's Progressive Matrices, a common intelligence test, involved:
- **Traditional Approach**: Extracting propositional representations and using engines on them.
- **New Approach**: Using analogical representations without propositional extraction.
- **Example**: ASTI (Affine and Set Transformation Inference) used affine and set transformations over analogical representations to solve problems, achieving high accuracy.
- **Further Development**: Keith McGregor developed a program using fractal representations, also successfully addressing Raven's test problems.
  
  ---
  
  This comprehensive summary captures all details from the lecture, covering visuospatial reasoning concepts, different views of reasoning, examples of AI systems utilizing visuospatial knowledge, and advancements in AI problem-solving approaches.
-
- Certainly! Here’s the second part of the lecture transformed into a structured format:
- ### Lecture Notes on Systems Thinking and Creativity in AI
- #### 614 - Systems Thinking: Introduction
- **AI Agents and Systems Thinking:**
	- AI agents must reason about the external world, which consists of various systems.
	- A system is made of heterogeneous interacting components leading to diverse processes.
	- Processes occur at different levels of abstraction and can be invisible.
	- Example: Ecosystems with physical, biological, and chemical processes.
	- Businesses consist of multiple interacting units (e.g., manufacturing, marketing).
- **Key Concepts:**
	- **Systems Thinking:** Ability to understand the invisible properties and complex behaviors of systems.
	- **Deriving Invisible Processes:** From visible structures.
- #### 615 - Systems Thinking: Connections
- **Relating Systems Thinking to Previous Topics:**
	- **Frames and Stories:** Capturing information about systems (e.g., political systems).
	- **Scripts:** Understanding complex systems (e.g., dining at a restaurant).
	- **Diagnosis:** Identifying faults in malfunctioning systems.
	- **Example:** Program debugging and ecological systems (e.g., drop in bee population due to insecticides).
- **Importance of Multiple Levels of Abstraction:**
	- Visible and invisible layers in complex systems.
	- Systems thinking helps understand these layers.
- #### 616 - Structure-Behavior-Function (SBF)
- **SBF Models in AI:**
	- Capture visible structures and invisible processes (behavior and function).
	- Example: Flashlight components and their functions.
- **Modeling Example:**
	- **Structure:** Button, bulb, battery.
	- **Behavior:** Electricity flow, conversion to light.
	- **Function:** Creating light.
- **Nested SBF Models:**
	- Flashlight circuit, bulb, etc.
	- Enable systems thinking in diagnosis and design.
- #### 617 - Design: Introduction
- **Design Thinking:**
	- Open-ended, under-constrained problems (e.g., designing a sustainable house).
	- **Problem-Solution Co-evolution:** Both problem and solution evolve together.
- #### 618 - Agents Doing Design
- **Configuration Design:**
	- Known components, finding arrangements, and assigning values.
	- Examples: Designing a chair, routine design tasks.
- **Creative Design:**
	- Unknown parts, innovative solutions.
	- Example: Flashlight circuit with increased lumens by using series-connected batteries.
	- **Learning from Design:** Design patterns (e.g., IDOL program learning design patterns).
- **Analogical Transfer:**
	- Applying design patterns across different domains (e.g., electrical circuits to water pumps).
- #### 619 - Creativity: Introduction
- **Creativity in AI:**
	- Creating AI agents that think and act creatively like humans.
	- **Defining Creativity:** Producing non-obvious, desirable products.
- #### 620 - Exercise: Defining Creativity I Question
- **Exercise Prompt:**
	- Reflect on what constitutes creativity.
- #### 621 - Exercise: Defining Creativity I Solution
- **Example Definition:**
	- Creativity involves producing a non-obvious, desirable product.
	- Discussion on different interpretations of creativity.
- #### 622 - Defining Creativity II
- **Novelty and Unexpectedness:**
	- Example: Creating a new soufflé recipe.
	- **Creative Processes:** Analogical reasoning, explanation-based learning, emergence, re-representation, serendipity.
- #### 623 - Exercise: Defining Creativity III Question
- **Exercise Prompt:**
	- Determine if AI agents performing specific tasks are creative.
- #### 624 - Exercise: Defining Creativity III Solution
- **Discussion on Creativity in AI:**
	- Challenges in defining AI creativity due to predictability and context dependence.
	- Different perspectives on whether AI can produce novel and unexpected results.
- #### 626 - Exercise: Defining Creativity IV Solution
- **Detailed Analysis:**
	- Consideration of algorithms producing novel outputs.
	- Context and situation influence on AI creativity.
	- Creativity defined by output versus process.
	- Encouragement to continue discussions on the forum.
	  
	  This structured format presents the lecture content systematically, making it easier to follow and understand the key concepts and discussions on systems thinking, design, and creativity in AI.
- ## 614 - Systems Thinking: Introduction
  
  AI agents must reason about the external world, which consists of systems with heterogeneous interacting components. These components and their interactions lead to various processes occurring at different levels of abstraction, some of which are invisible. For example, in an ecosystem, physical, biological, and chemical processes interact at multiple levels, often invisibly but influencing each other. Similarly, businesses comprise interacting units like manufacturing, marketing, and delivery, each describable at different abstraction levels—from individuals to full organizations. AI agents need systems thinking to understand these invisible properties and complex behaviors, deriving invisible processes from visible structures.
- ## 615 - Systems Thinking: Connections
  
  Systems thinking connects to several topics covered in the course:
- **Frames and Stories**: Frames help understand stories, which capture information about systems, such as a political system where earthquakes occur due to geological faults.
- **Scripts**: A dining experience at a restaurant is a complex system with multiple components, relationships, interactions, and processes.
- **Diagnosis**: Identifying a fault in a malfunctioning system involves understanding the interaction of components and complex behaviors. For instance, diagnosing a drop in bee populations involves inferring the poisoning process from the visible drop in population and the rise in pesticide use.
  
  In complex systems, multiple levels of abstraction exist, some visible and some invisible. Systems thinking helps understand the invisible levels.
- ## 616 - Structure-Behavior-Function
  
  AI uses representations to capture both visible structures and invisible levels like behavior and function. These models are called Structure-Behavior-Function (SBF) models. For example, a flashlight has visible components (button, bulb, battery) and invisible processes (electricity flowing and converting to light). SBF models capture the structure (components and their connections), behavior (electricity flow and conversion to light), and function (producing light) of systems. These models are nested, allowing for multiple levels of abstraction, such as modeling the flashlight circuit and then the lightbulb itself. SBF models enable systems thinking in diagnosis and design of complex systems.
- ## 617 - Design: Introduction
  
  Design thinking involves addressing ill-defined, under-constrained, open-ended problems. For example, designing a sustainable house involves co-evolving the problem and solution, as the problem evolves alongside the solution. Design thinking is not just about finding a solution but also about defining and refining the problem itself.
- ## 618 - Agents Doing Design
  
  Configuration design involves known components and arranging them, assigning values to variables to meet constraints. AI systems in industry use methods like brand refinement, case-based reasoning, model-based reasoning, and rule-based reasoning for configuration design. Creative design involves unknown parts and requires innovative approaches. For instance, designing a flashlight to produce more light might involve using two 1.5-volt batteries in series when a 3-volt battery is unavailable. The AI program IDOL, created by Sam at Georgia Tech, learned design patterns like cascading (replicating behaviors) and applied them across domains, such as electrical circuits to water pumps. This analogical transfer of design patterns demonstrates creative design.
- ## 619 - Creativity: Introduction
  
  Humans are naturally creative, dealing with problems daily. Knowledge-based AI aims to create AI agents that think and act like humans, including being creative. Defining creativity is challenging, similar to defining intelligence.
- ## 620 - Exercise: Defining Creativity I Question
  
  To build creative AI agents, we must define creativity. Students are asked to define creativity and post their answers in the class forum.
- ## 621 - Exercise: Defining Creativity I Solution
  
  Creativity is defined as producing a non-obvious, desirable product. The product must be wanted and not an obvious solution. However, definitions may vary; some may not consider a product necessary, while others may not include desirability. The discussion continues on the forum.
- ## 622 - Defining Creativity II
  
  Novelty involves newness, while unexpectedness involves something non-obvious or surprising. For example, making a soufflé for 20 friends is novel if you’ve never done it before. Creating a dramatically different soufflé recipe is unexpected. Creativity involves both product and process. Analogical reasoning, emergence, re-representation, and serendipity are fundamental creative processes. For example, drawing three lines to form a triangle (emergence), re-representing the atomic structure to map it to the solar system (re-representation), or discovering Velcro while solving a different problem (serendipity).
- ## 623 - Exercise: Defining Creativity III Question
  
  Students evaluate tasks from the course, marking if they think the agent performing the task was creative.
- ## 624 - Exercise: Defining Creativity III Solution
  
  None of the tasks were marked as creative. AI tasks often involve predictable algorithms, making outputs non-creative. However, this is debatable; algorithms for open-ended problems can produce novel results, and creativity can be defined in terms of both output and process.
- ## 626 - Exercise: Defining Creativity IV Solution
  
  Reviewing task choices:
- **First Choice**: An algorithm’s output can be novel in open-ended problems.
- **Second Choice**: Output depends on input, methods, and context, making it sometimes unexpected.
- **Third Choice**: Creativity can be defined by output without knowing the process.
  The fourth answer is preferred, but discussion continues on the forum.
  
  This detailed summary includes more explanation and examples to elaborate on each part of the lecture, providing a comprehensive understanding of the concepts covered.