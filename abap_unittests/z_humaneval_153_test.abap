CLASS ltcl_test_strongest_extension DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING,
      test_case_8 FOR TESTING,
      test_case_9 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_strongest_extension IMPLEMENTATION.

  METHOD test_case_1.
    DATA: extensions TYPE string_table.
    extensions = VALUE #( ( |tEN| ) ( |niNE| ) ( |eIGHt8OKe| ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_153=>strongest_extension( class_name = 'Watashi' extensions = extensions )
      exp = 'Watashi.eIGHt8OKe' ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: extensions TYPE string_table.
    extensions = VALUE #( ( |nani| ) ( |NazeDa| ) ( |YEs.WeCaNe| ) ( |32145tggg| ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_153=>strongest_extension( class_name = 'Boku123' extensions = extensions )
      exp = 'Boku123.YEs.WeCaNe' ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: extensions TYPE string_table.
    extensions = VALUE #( ( |t| ) ( |eMptY| ) ( |nothing| ) ( |zeR00| ) ( |NuLl__| ) ( |123NoooneB321| ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_153=>strongest_extension( class_name = '__YESIMHERE' extensions = extensions )
      exp = '__YESIMHERE.NuLl__' ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: extensions TYPE string_table.
    extensions = VALUE #( ( |Ta| ) ( |TAR| ) ( |t234An| ) ( |cosSo| ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_153=>strongest_extension( class_name = 'K' extensions = extensions )
      exp = 'K.TAR' ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: extensions TYPE string_table.
    extensions = VALUE #( ( |Tab| ) ( |123| ) ( |781345| ) ( |-_-| ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_153=>strongest_extension( class_name = '__HAHA' extensions = extensions )
      exp = '__HAHA.123' ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: extensions TYPE string_table.
    extensions = VALUE #( ( |HhAas| ) ( |okIWILL123| ) ( |WorkOut| ) ( |Fails| ) ( |-_-| ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_153=>strongest_extension( class_name = 'YameRore' extensions = extensions )
      exp = 'YameRore.okIWILL123' ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: extensions TYPE string_table.
    extensions = VALUE #( ( |Die| ) ( |NowW| ) ( |Wow| ) ( |WoW| ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_153=>strongest_extension( class_name = 'finNNalLLly' extensions = extensions )
      exp = 'finNNalLLly.WoW' ).
  ENDMETHOD.

  METHOD test_case_8.
    DATA: extensions TYPE string_table.
    extensions = VALUE #( ( |Bb| ) ( |91245| ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_153=>strongest_extension( class_name = '_' extensions = extensions )
      exp = '_.Bb' ).
  ENDMETHOD.

  METHOD test_case_9.
    DATA: extensions TYPE string_table.
    extensions = VALUE #( ( |671235| ) ( |Bb| ) ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_153=>strongest_extension( class_name = 'Sp' extensions = extensions )
      exp = 'Sp.671235' ).
  ENDMETHOD.

ENDCLASS.