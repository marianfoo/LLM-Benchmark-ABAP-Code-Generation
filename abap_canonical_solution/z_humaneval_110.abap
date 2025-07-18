CLASS z_humaneval_110 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    TYPES: BEGIN OF ty_result,
             value TYPE string,
           END OF ty_result.

    CLASS-METHODS: exchange
      IMPORTING
        lst1          TYPE int4_table
        lst2          TYPE int4_table
      RETURNING
        VALUE(result) TYPE string.

  PRIVATE SECTION.
ENDCLASS.

CLASS z_humaneval_110 IMPLEMENTATION.
  METHOD exchange.
    DATA: odd  TYPE i VALUE 0,
          even TYPE i VALUE 0,
          temp TYPE i.

    LOOP AT lst1 INTO temp.
      IF temp MOD 2 = 1.
        odd = odd + 1.
      ENDIF.
    ENDLOOP.

    LOOP AT lst2 INTO temp.
      IF temp MOD 2 = 0.
        even = even + 1.
      ENDIF.
    ENDLOOP.

    IF even >= odd.
      result = 'YES'.
    ELSE.
      result = 'NO'.
    ENDIF.
  ENDMETHOD.
ENDCLASS.