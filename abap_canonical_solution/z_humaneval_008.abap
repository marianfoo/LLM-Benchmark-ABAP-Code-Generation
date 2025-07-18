CLASS z_humaneval_008 DEFINITION
  FINAL
  PUBLIC
  CREATE PUBLIC.

  PUBLIC SECTION.
    " Option 1: Using a structure to combine both values
    TYPES: BEGIN OF ty_result,
             sum_value  TYPE i,
             prod_value TYPE i,
           END OF ty_result.

    CLASS-METHODS sum_product
      IMPORTING
        VALUE(numbers) TYPE int4_table
      RETURNING
        VALUE(result)  TYPE ty_result.

ENDCLASS.

CLASS z_humaneval_008 IMPLEMENTATION.

  METHOD sum_product.

    result-sum_value = 0.
    result-prod_value = 1.

    IF lines( numbers ) = 0.
      RETURN.
    ENDIF.

    LOOP AT numbers INTO DATA(n).
      result-sum_value = result-sum_value + n.
      result-prod_value = result-prod_value * n.
    ENDLOOP.

  ENDMETHOD.

ENDCLASS.