# General Commands:

`git status` - Shows your current branch and lists any differences between your local files and the online git repo

`git checkout <branch>` - Switch from current branch to \<branch\>

`git pull` - Pull any changes from the git repo.

`git branch <branch name>` - Creates a new branch with the name \<branch name\>

`git checkout -b <branch name>` - Create a new branch with the name \<branch name\> and 'checks out' that branch

`git push -u origin <branch name>` - Pushes your current branch to the online repo

`git merge <branch>` - Merges \<branch\> into your current branch

`git branch` - lists your branches and highlights the current branch.

`git add <file>` - 'Stage' a file so that it may be commited

`git add -A` - Stages all your local changes (Might not want to do this if you have files that you don't want to commit)

`git commit -m "<message>"` - Commit your changes

# Creating a new branch:
  `git status`
  
  `git checkout -b <branch name>`
  
  `git push -u origin <branch name>`

# Creating a commit:
`git add <file>`

`git commit -m "<message>"`

`git push`

# Merging another branch into your branch:


`git checkout <other branch to merge>`

`git pull`

`git checkout <your branch>`

`git merge <other branch to merge>`


# Merging Main into your branch:

Do this when a new feature gets merged into main so that you can get the changes in the branch you are working on

`git checkout main`

`git pull`

`git checkout <branch>`

`git merge main`

# To Test Out A Pull Request

When someone makes a pull request, you will probably want to test out their changes locally before it gets merged into the main branch. To do this, do the following:

Assuming you are starting from your working branch:
1. `git status` - If it says 'nothing to commit, working tree clean', you are good and you can skip to step 4. Otherwise you will want to stash or commit your current changes. I may go into how to use git stash later, but for now, we will just commit everything.
2. `git add -A`
3. `git commit -m "WIP"` - When I am in the middle of doing something, I make my commit message WIP - Work In Progress
5. `git pull` - brings down the new branches, so you can access them locally
6. `git checkout <branch_name>` - The branch name will be in the pull request, right underneath the title, where it says '<user> wants to merge n commits into main from \<branch_name\>'

Now you can run the appropriate scripts to test out the code in the pull request. When you are done, make sure you switch back to your working branch.

7. `git checkout <my branch>`

This process preserves your changes and allows you to test out someone else's code without affecting any of your own code.

