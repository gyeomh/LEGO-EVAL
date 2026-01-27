from ai2thor.controller import Controller

THOR_COMMIT_ID = "3213d486cd09bcbafce33561997355983bdf8d1a"

c = Controller(
            commit_id=THOR_COMMIT_ID,
            agentMode="default",
            makeAgentsVisible=False,
            visibilityDistance=1.5,
            scene='Procedural',
            width=1200,
            height=1200,
            fieldOfView=90,
            x_org=":4", # set this as your display number. 
            )
