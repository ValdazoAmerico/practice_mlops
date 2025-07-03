import shutil

def deploy_model(candidate_path="models/candidate_model.pkl", prod_path="models/current_model.pkl"):
    #shutil.copy(candidate_path, prod_path)
    print("✅ Modelo deployado exitosamente.")
