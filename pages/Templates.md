- Daily Plan
  template:: Daily Board
- Daily Plan Morning
  template:: Daily Board Morning
	- {{renderer :kanban_6594fd13-57e4-45d3-bc40-57897642c1e3}}
		- tasks
			- TODO Programming read 20 min.
			- TODO morning yoga
			- TODO mid day ab roller wrist training
			- TODO cardio
			- TODO supplementry training
			- TODO prayers
			- TODO math refresh 10min
- Daily plan evening
  template:: Daily Board evening
	- {{renderer :kanban_65c018ed-f12f-434c-9a66-2f59fe02ad55}}
		- tasks
			- TODO prayers
			- TODO 1.5 hr of study.
			- TODO Study Read 40 min.
			- TODO Weight lifting 45 min.
			-
- Working Day
-
- ### "More Journal Templates" plugin >
  comment:: [English document](https://github.com/YU000jp/logseq-plugin-weekdays-and-weekends/wiki/English-document)
	- #### Journal-template config >
	  template:: Journal
	  template-including-parent:: false
	  comment:: Replace `:default-templates {:journals "Journal"}` in the "config.edn" file. Rendering occurs when it is loaded as a diary template.
	  background-color:: yellow
		- {{renderer :Weekdays, Main-Template, Sun&Mon&Tue&Wed&Thu&Fri}}
		  {{renderer :Weekdays, Weekends-Template, Sat&Sun}}
		-
	- #### Main-Template > 
	  background-color:: gray
	  template:: Main-Template
	  template-including-parent:: true
		- #### AM
			- {{renderer :kanban_6594fd13-57e4-45d3-bc40-57897642c1e3}}
				- tasks
					- TODO Programming read 20 min.
					- TODO morning yoga
					- TODO mid day ab roller wrist training
			- TODO 1.5 hr of study.
			  :LOGBOOK:
			  CLOCK: [2024-01-03 Wed 07:47:04]--[2024-01-03 Wed 07:47:09] =>  00:00:05
			  :END:
					- TODO cardio
					- TODO supplementry training
					- TODO prayers
					- TODO math refresh 10min
			-
		- #### PM
		  id:: 65bf631a-7c1c-4bfc-920d-b278aeb3dfe4
			- {{renderer :kanban_65c018ed-f12f-434c-9a66-2f59fe02ad55}}
				- tasks
					- TODO prayers
					- TODO 1.5 hr of study.
					- TODO Study Read 40 min.
					- TODO Weight lifting 45 min.
					-
			-
	- #### Sub-Template > 
	  template:: Sub-Template
	  template-including-parent:: false
	  background-color:: gray
		- #### AM
			-
			-
		- #### PM
			-
			-
	- #### Weekends-Template > 
	  template:: Weekends-Template
	  template-including-parent:: false
	  background-color:: gray
		- #### AM
			-
			-
		- #### PM
			-
	- #### Holidays-Template > 
	  template:: Holidays-Template
	  template-including-parent:: false
	  background-color:: gray
		- #### AM
			-
			-
		- #### PM
			-
-
-
-
- ### "More Journal Templates" plugin >
  comment:: [English document](https://github.com/YU000jp/logseq-plugin-weekdays-and-weekends/wiki/English-document)
	- #### Journal-template config >
	  template:: Journal
	  template-including-parent:: false
	  comment:: Replace `:default-templates {:journals "Journal"}` in the "config.edn" file. Rendering occurs when it is loaded as a diary template.
	  background-color:: yellow
		- {{renderer :Weekdays, Mon-Template, Mon}}
		  {{renderer :Weekdays, Tue-Template, Tue}}
		  {{renderer :Weekdays, Wed-Template, Wed}}
		  {{renderer :Weekdays, Thu-Template, Thu}}
		  {{renderer :Weekdays, Fri-Template, Fri}}
		  {{renderer :Weekdays, Sat-Template, Sat}}
		  {{renderer :Weekdays, Sun-Template, Sun}}
		-
	- #### Mon-Template > 
	  background-color:: gray
	  template:: Mon-Template
	  template-including-parent:: false
		- #### AM
			-
		- #### PM
			-
		-
	- #### Tue-Template > 
	  template:: Tue-Template
	  template-including-parent:: false
	  background-color:: gray
		- #### AM
			-
			-
		- #### PM
			-
			-
	- #### Wed-Template > 
	  template:: Wed-Template
	  template-including-parent:: false
	  background-color:: gray
		- #### AM
			-
			-
		- #### PM
			-
			-
	- #### Thu-Template > 
	  template:: Thu-Template
	  template-including-parent:: false
	  background-color:: gray
		- #### AM
			-
			-
		- #### PM
			-
			-
	- #### Fri-Template > 
	  template:: Fri-Template
	  template-including-parent:: false
	  background-color:: gray
		- #### AM
			-
			-
		- #### PM
			-
			-
	- #### Sat-Template > 
	  template:: Sat-Template
	  template-including-parent:: false
	  background-color:: gray
		- #### AM
			-
			-
		- #### PM
			-
			-
	- #### Sun-Template > 
	  template:: Sun-Template
	  template-including-parent:: false
	  background-color:: gray
		- #### AM
			-
			-
		- #### PM
			-
			-
	- #### Main-Template > 
	  template:: Main-Template
	  template-including-parent:: false
	  background-color:: gray
		- #### AM
			-
			-
		- #### PM
			-
			-
	- #### Sub-Template > 
	  template:: Sub-Template
	  template-including-parent:: false
	  background-color:: gray
		- #### AM
			-
			-
		- #### PM
			-
			-
	- #### Weekends-Template > 
	  template:: Weekends-Template
	  template-including-parent:: false
	  background-color:: gray
		- #### AM
			-
			-
		- #### PM
			-
			-
	- #### Holidays-Template > 
	  template:: Holidays-Template
	  template-including-parent:: false
	  background-color:: gray
		- #### AM
			-
			-
		- #### PM
			-
			-