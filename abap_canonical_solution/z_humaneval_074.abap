CLASS z_humaneval_074 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: total_match
      IMPORTING
        lst1          TYPE string_table
        lst2          TYPE string_table
      RETURNING
        VALUE(result) TYPE string_table.

ENDCLASS.

CLASS z_humaneval_074 IMPLEMENTATION.
  METHOD total_match.
    DATA: l1 TYPE i,
          l2 TYPE i,
          st TYPE string.

    l1 = 0.
    LOOP AT lst1 INTO st.
      l1 = l1 + strlen( st ).
    ENDLOOP.

    l2 = 0.
    LOOP AT lst2 INTO st.
      l2 = l2 + strlen( st ).
    ENDLOOP.

    IF l1 <= l2.
      result = lst1.
    ELSE.
      result = lst2.
    ENDIF.
  ENDMETHOD.
ENDCLASS.