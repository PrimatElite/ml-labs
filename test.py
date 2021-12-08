import time

from src.checker.tester import Tester


start = time.time()
tester = Tester('../placer_dataset', 'intelligent_placer_lib/default_config.yml')
tester.run()
end = time.time()
print(end - start)
