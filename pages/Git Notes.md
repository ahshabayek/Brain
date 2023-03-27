- git config
	- `git config --global core.editor "code --wait"`
	- ```
	  git config --global color.branch auto
	  git config --global color.diff auto
	  git config --global color.interactive auto
	  git config --global color.status auto
	  ```
	- id:: 641711f8-1508-4363-bfe5-fc95cfad39ee
	  ```
	  parse_git_branch() {
	       git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/(\1)/'
	  }
	  export PS1="\u@\h \[\e[32m\]\w \[\e[91m\]\$(parse_git_branch)\[\e[00m\]$ "
	  ```
-
-