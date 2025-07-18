CLASS z_humaneval_107 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    TYPES: BEGIN OF ty_result,
             even_count TYPE i,
             odd_count  TYPE i,
           END OF ty_result.

    CLASS-METHODS: even_odd_palindrome
      IMPORTING
        VALUE(n)      TYPE i
      RETURNING
        VALUE(result) TYPE ty_result.

  PRIVATE SECTION.
    CLASS-METHODS: is_palindrome
      IMPORTING
        VALUE(num)    TYPE i
      RETURNING
        VALUE(is_pal) TYPE abap_bool.

ENDCLASS.

CLASS z_humaneval_107 IMPLEMENTATION.
  METHOD even_odd_palindrome.
    DATA: even_palindrome_count TYPE i VALUE 0,
          odd_palindrome_count  TYPE i VALUE 0,
          i                     TYPE i.

    DO n TIMES.
      i = sy-index.
      IF i MOD 2 = 1 AND is_palindrome( i ) = abap_true.
        odd_palindrome_count = odd_palindrome_count + 1.
      ELSEIF i MOD 2 = 0 AND is_palindrome( i ) = abap_true.
        even_palindrome_count = even_palindrome_count + 1.
      ENDIF.
    ENDDO.

    result-even_count = even_palindrome_count.
    result-odd_count = odd_palindrome_count.
  ENDMETHOD.

  METHOD is_palindrome.
    DATA: num_str TYPE string,
          rev_str TYPE string,
          len     TYPE i,
          i       TYPE i.

    num_str = |{ num }|.
    len = strlen( num_str ).
    rev_str = ''.

    DO len TIMES.
      i = len - sy-index.
      rev_str = rev_str && num_str+i(1).
    ENDDO.

    IF num_str = rev_str.
      is_pal = abap_true.
    ELSE.
      is_pal = abap_false.
    ENDIF.
  ENDMETHOD.

ENDCLASS.