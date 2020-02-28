# Reinforcement Learning for Robot-Task-Combinations
Catchy name pending...

## Contribution guidelines

We use the [gitlab issue system](https://git.uni-wuppertal.de/scheiderer/robot-task-rl/issues) to organize this project.

Issues are organized by tags which are categorized into Type, Status and Priority.

### Workflow

Your workflow for starting a new task should look like this:
1. Decide which issue you want to work on
    * Check if you are assigned to any issues
    * If you want to work on an issue you are not assigned to yet, 
    assign yourself to signal to others that you are working on the issue.
3. Add the label "Status: In Progress" to signal that the issue is being worked on
4. Create a new branch from master
    * Use the "Create Branch" button inside the issue.
    * You can change the branch name to a shorter (**but still meaningful**) name, but make sure that the branch name starts with the issue number.
5. Checkout the issue branch and fix the issue.
    * **Make sure to commit regularly.** Thats what git is for.
    * [Use meaningful commit messages!](https://www.freecodecamp.org/news/writing-good-commit-messages-a-practical-guide/)
    * If you need input / help, don't hesitate to use the comment section of the issue.
6. Once you are satisfied with your code, merge the master branch into your branch.
    * Fix potential merge conflicts
    * Make sure your code still runs as intended
7. Change the status of the issue to "Status: Ready for Review"
8. Create a merge request for your branch.
    * Use the "Create Merge Request" button inside the issue.
9. Repeat from 1.

Once you created the merge request, your code will be reviewed. 
If everything is fine your changes will be merged into master. 
If there are some issues you will be notified via the issue discussion.

In some occasions, you are not the first to work on an issue.
Make sure to check if other people are assigned to the issue, if there are already related branches, etc.
If you are unsure what part of the issue is still open, use of the comment section of the issue.

### Creating a new issue

Feel free to create a new issues if necessary.

When creating an issue, choose a meaningful title and a description.
Also assign the correct `Type` tag.