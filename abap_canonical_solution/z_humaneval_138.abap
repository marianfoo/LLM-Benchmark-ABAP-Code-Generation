CLASS z_humaneval_138 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: is_equal_to_sum_even
      IMPORTING
        VALUE(n)      TYPE i
      RETURNING
        VALUE(result) TYPE abap_bool.

ENDCLASS.

CLASS z_humaneval_138 IMPLEMENTATION.

  METHOD is_equal_to_sum_even.
    result = COND #( WHEN n MOD 2 = 0 AND n >= 8 THEN abap_true ELSE abap_false ).
  ENDMETHOD.

ENDCLASS.