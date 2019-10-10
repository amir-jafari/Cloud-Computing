 function list_iter (t)
      local i = 0
      local n = table.getn(t)
      return function ()
               i = i + 1
               if i <= n then return t[i] end
             end
    end

    t = {10, 20, 30}
    iter = list_iter(t)    -- creates the iterator
    while true do
      local element = iter()   -- calls the iterator
      if element == nil then break end
      print(element)
    end