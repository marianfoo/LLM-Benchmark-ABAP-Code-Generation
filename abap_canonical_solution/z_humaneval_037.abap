CLASS z_humaneval_037 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: sort_even
      IMPORTING
        VALUE(l)      TYPE int4_table
      RETURNING
        VALUE(result) TYPE int4_table.

  PRIVATE SECTION.

ENDCLASS.

CLASS z_humaneval_037 IMPLEMENTATION.

  METHOD sort_even.
    DATA: evens    TYPE int4_table,
          odds     TYPE int4_table,
          i        TYPE i,
          even_idx TYPE i,
          odd_idx  TYPE i.

    i = 0.
    LOOP AT l INTO DATA(value).
      IF i MOD 2 = 0.
        evens = VALUE #( BASE evens ( value ) ).
      ELSE.
        odds = VALUE #( BASE odds ( value ) ).
      ENDIF.
      i = i + 1.
    ENDLOOP.

    SORT evens.

    even_idx = 1.
    odd_idx = 1.
    WHILE even_idx <= lines( evens ).
      result = VALUE #( BASE result ( evens[ even_idx ] ) ).
      IF odd_idx <= lines( odds ).
        result = VALUE #( BASE result ( odds[ odd_idx ] ) ).
        odd_idx = odd_idx + 1.
      ENDIF.
      even_idx = even_idx + 1.
    ENDWHILE.

  ENDMETHOD.

ENDCLASS.